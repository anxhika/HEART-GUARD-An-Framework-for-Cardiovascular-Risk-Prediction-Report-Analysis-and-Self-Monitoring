import json
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.contrib.auth.decorators import login_required

from .models import ChatSession, ChatMessage
from .rag.pipeline import run_pipeline
from .rag.emergency_detector import is_emergency
from .context.prediction_context import build_prediction_context
from .context.report_context import build_report_context


def _get_or_create_session(user, session_id=None):
    """Get an existing session or create a new one."""
    if session_id:
        try:
            return ChatSession.objects.get(pk=session_id, user=user)
        except ChatSession.DoesNotExist:
            pass
    return ChatSession.objects.create(user=user)


def _get_history(session, max_messages=None):
    """Return the last N messages as a list of dicts."""
    max_messages = max_messages or settings.CHATBOT_MAX_HISTORY
    messages = session.messages.order_by('-timestamp')[:max_messages]
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]


@login_required
@require_http_methods(["POST"])
def chat_api(request):
    """
    Main chat endpoint. Returns a Server-Sent Events (SSE) streaming response.
    POST body JSON: {"message": str, "session_id": int|null}
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    user_message = body.get("message", "").strip()
    session_id = body.get("session_id")

    if not user_message:
        return JsonResponse({"error": "Message cannot be empty."}, status=400)
    if len(user_message) > 2000:
        return JsonResponse({"error": "Message too long (max 2000 characters)."}, status=400)

    # Get or create session
    session = _get_or_create_session(request.user, session_id)

    # Save user message
    ChatMessage.objects.create(session=session, role='user', content=user_message)
    session.save()  # update updated_at

    # Build context
    history = _get_history(session, max_messages=settings.CHATBOT_MAX_HISTORY)
    prediction_context = build_prediction_context(request.user)
    report_context = build_report_context(request.user)

    # Run pipeline
    result = run_pipeline(
        user_query=user_message,
        prediction_context=prediction_context,
        report_context=report_context,
        history=history[:-1],  # exclude the message we just added
    )

    # Handle errors
    if result["error"]:
        return JsonResponse({"error": result["error"], "session_id": session.pk}, status=503)

    sources = result["sources"]

    # Emergency: return immediately as plain JSON (no streaming)
    if result["is_emergency"]:
        response_text = result["response"]
        ChatMessage.objects.create(session=session, role='assistant', content=response_text, sources=[])
        return JsonResponse({
            "session_id": session.pk,
            "response": response_text,
            "sources": [],
            "is_emergency": True,
            "streaming": False,
        })

    # Normal: stream tokens via SSE
    def event_stream():
        full_response = []
        yield f"data: {{\"session_id\": {session.pk}, \"sources\": {json.dumps(sources)}}}\n\n"
        try:
            for token in result["response"]:
                full_response.append(token)
                # Escape the token for JSON
                escaped = json.dumps(token)
                yield f"data: {{\"token\": {escaped}}}\n\n"
        except Exception as e:
            yield f"data: {{\"error\": {json.dumps(str(e))}}}\n\n"
        finally:
            # Save complete assistant message after streaming finishes
            complete_text = "".join(full_response)
            if complete_text.strip():
                ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content=complete_text,
                    sources=sources,
                )
            yield "data: [DONE]\n\n"

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


@login_required
@require_http_methods(["GET"])
def chat_history(request):
    """Returns all messages for a session. GET ?session_id=<id>"""
    session_id = request.GET.get("session_id")
    if not session_id:
        return JsonResponse({"error": "session_id required"}, status=400)
    try:
        session = ChatSession.objects.get(pk=session_id, user=request.user)
    except ChatSession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)
    messages = [
        {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat(), "sources": m.sources}
        for m in session.messages.all()
    ]
    return JsonResponse({"session_id": session.pk, "messages": messages})


@login_required
@require_http_methods(["POST"])
def new_session(request):
    """Creates a fresh chat session. Returns the new session ID."""
    session = ChatSession.objects.create(user=request.user)
    return JsonResponse({"session_id": session.pk})


@login_required
@require_http_methods(["GET"])
def session_list(request):
    """
    Returns all chat sessions for the logged-in user, most recent first.
    Each session includes its id, timestamps, and a preview of the first user message.
    """
    sessions = ChatSession.objects.filter(user=request.user).order_by('-updated_at')[:30]
    data = []
    for s in sessions:
        first_msg = s.messages.filter(role='user').order_by('timestamp').first()
        preview = (first_msg.content[:60] + '…') if first_msg and len(first_msg.content) > 60 else (first_msg.content if first_msg else 'Empty chat')
        data.append({
            "session_id": s.pk,
            "preview": preview,
            "updated_at": s.updated_at.isoformat(),
            "created_at": s.created_at.isoformat(),
            "message_count": s.messages.count(),
        })
    return JsonResponse({"sessions": data})
