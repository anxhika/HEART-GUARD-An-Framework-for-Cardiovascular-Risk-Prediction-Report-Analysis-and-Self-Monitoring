"""
Django management command to load, chunk, embed and store medical PDF guidelines.
Usage: python manage.py ingest_guidelines
"""
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from chatbot.rag.pdf_loader import load_and_chunk_all
from chatbot.rag.vector_store import add_documents, collection_count


class Command(BaseCommand):
    help = 'Ingest medical guideline PDFs into the ChromaDB vector store.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-ingestion even if documents already exist.',
        )

    def handle(self, *args, **options):
        force = options.get('force', False)
        guidelines_dir = settings.CHATBOT_GUIDELINES_DIR

        if not os.path.exists(guidelines_dir):
            self.stderr.write(self.style.ERROR(f"Guidelines directory not found: {guidelines_dir}"))
            return

        pdf_files = [f for f in os.listdir(guidelines_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            self.stderr.write(self.style.WARNING("No PDF files found in guidelines directory."))
            return

        existing_count = collection_count()
        if existing_count > 0 and not force:
            self.stdout.write(
                self.style.WARNING(
                    f"{existing_count} chunks already exist in ChromaDB. "
                    f"Use --force to re-ingest."
                )
            )
            return

        self.stdout.write(f"Found {len(pdf_files)} PDF file(s): {', '.join(pdf_files)}")
        self.stdout.write("Loading and chunking PDFs...")
        chunks = load_and_chunk_all()

        if not chunks:
            self.stderr.write(self.style.ERROR("No chunks were created. Check PDF content."))
            return

        self.stdout.write(f"Embedding {len(chunks)} chunks into ChromaDB (this may take a few minutes)...")
        add_documents(chunks)

        final_count = collection_count()
        self.stdout.write(self.style.SUCCESS(
            f"Done! {final_count} chunks stored in ChromaDB collection '{settings.CHATBOT_CHROMA_COLLECTION}'."
        ))
