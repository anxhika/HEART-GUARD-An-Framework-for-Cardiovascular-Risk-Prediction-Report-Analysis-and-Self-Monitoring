
from django.shortcuts import render, redirect
from .forms import Parameters
from .regressor import predict_heart_disease
from django.contrib import messages
from django.contrib.auth.models import User, auth
from .models import HeartData, DoctorHospital
from django.core.mail import send_mail


# Predicts results for users who are not authenticated
def quickpredict(request):

    if request.method == 'POST':

        form = Parameters(request.POST)

        if form.is_valid():

            age = form.cleaned_data['age']
            sex = form.cleaned_data['sex']
            cp = form.cleaned_data['cp']
            trestbps = form.cleaned_data['trestbps']
            chol = form.cleaned_data['chol']
            fbs = form.cleaned_data['fbs']
            restcg = form.cleaned_data['restcg']
            thalach = form.cleaned_data['thalach']
            exang = form.cleaned_data['exang']
            oldpeak = form.cleaned_data['oldpeak']
            slope = form.cleaned_data['slope']
            ca = form.cleaned_data['ca']
            thal = form.cleaned_data['thal']

            # Prepare input data in the same order as the CSV columns
            input_data = [
                age, sex, cp, trestbps, chol, fbs,
                restcg, thalach, exang, oldpeak, slope, ca, thal
            ]

            # Get prediction using the updated model
            output, output1 = predict_heart_disease(input_data)

            prob_pct = f"{output1 * 100:.2f}%"

            print("-----------------------------------------")
            print("Prediction:", output)
            print("Probability:", prob_pct)

            return render(
                request,
                'output.html',
                {'output1': prob_pct, 'probability': output1}
            )

        else:
            print('The form was not valid.')
            return redirect('/')

    else:
        form = Parameters()
        return render(request, 'quickpredict.html', {'form': form})


# Home page for authenticated and non-authenticated users
def index(request):

    if request.user.is_authenticated:

        if request.method == 'POST':

            form = Parameters(request.POST)

            if form.is_valid():

                age = form.cleaned_data['age']
                sex = form.cleaned_data['sex']
                cp = form.cleaned_data['cp']
                trestbps = form.cleaned_data['trestbps']
                chol = form.cleaned_data['chol']
                fbs = form.cleaned_data['fbs']
                restcg = form.cleaned_data['restcg']
                thalach = form.cleaned_data['thalach']
                exang = form.cleaned_data['exang']
                oldpeak = form.cleaned_data['oldpeak']
                slope = form.cleaned_data['slope']
                ca = form.cleaned_data['ca']
                thal = form.cleaned_data['thal']

                # Prepare input data
                input_data = [
                    age, sex, cp, trestbps, chol, fbs,
                    restcg, thalach, exang, oldpeak, slope, ca, thal
                ]

                # Get prediction
                output, output1 = predict_heart_disease(input_data)

                danger = 'high' if output == 1 else 'low'
                prob_pct = f"{output1 * 100:.2f}%"

                print("-----------------------------------------")
                print("Prediction:", output)
                print("Probability:", prob_pct)

                # Save prediction details in database
                saved_data = HeartData(
                    age=age,
                    sex=sex,
                    cp=cp,
                    trestbps=trestbps,
                    chol=chol,
                    fbs=fbs,
                    restcg=restcg,
                    thalach=thalach,
                    exang=exang,
                    oldpeak=oldpeak,
                    slope=slope,
                    ca=ca,
                    thal=thal,
                    owner=request.user,
                    probability=round(output1, 4)
                )

                saved_data.save()

                return render(
                    request,
                    'output2.html',
                    {
                        'output1': prob_pct,
                        'danger': danger,
                        'probability': output1
                    }
                )

            else:
                messages.error(request, 'Please enter valid details.')

        form = Parameters()
        return render(request, 'user.html', {'form': form})

    return render(request, 'index.html')


# Displays previous records of authenticated users
def record(request):

    if request.user.is_authenticated:
        record_data = HeartData.objects.filter(owner=request.user)
        return render(request, 'record.html', {'record_data': record_data})

    return redirect('/')


# Displays previous prediction results in detail
def heartdetail(request):

    if request.user.is_authenticated:
        record_data = HeartData.objects.filter(owner=request.user)
        return render(request, 'heartdetail.html', {'record_data': record_data})

    return redirect('/')


# Displays symptoms of heart disease
def symptoms(request):

    if request.user.is_authenticated:
        return render(request, 'symptoms.html')

    return redirect('/')


# Displays prevention information
def prevention(request):

    if request.user.is_authenticated:
        return render(request, 'prevention.html')

    return redirect('/')


# Displays doctors and hospitals
def doctorhospital(request):

    if request.user.is_authenticated:
        datas = DoctorHospital.objects.all()
        return render(
            request,
            'doctorshospitals.html',
            {'datas': datas}
        )

    return redirect('/')


# Contact page and email functionality
def contact(request):

    if request.method == 'POST':

        name = request.POST.get('name')
        email = request.POST.get('email')
        title = request.POST.get('title1')
        message = request.POST.get('message')

        send_mail(
            title,
            message + '\n' + 'From: ' + name + '\n' + 'Email: ' + email,
            from_email=email,
            recipient_list=['focusus1@gmail.com']
        )

        messages.success(request, 'Your message has been sent.')

    return render(request, 'contact.html')


# About page
def about(request):
    return render(request, 'about.html')


# User sign in
def signin(request):

    if request.method == 'POST':

        username = request.POST.get('username')
        password = request.POST.get('password')

        user = auth.authenticate(
            username=username,
            password=password
        )

        if user is not None:
            auth.login(request, user)
            return redirect('/')

        else:
            messages.warning(request, 'Invalid Credentials')
            return redirect('signin')

    return render(request, 'signin.html')


# User sign up
def signup(request):

    if request.method == 'POST':

        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')

        if User.objects.filter(username=username).exists():

            messages.info(request, 'Username taken')
            return redirect('signup')

        elif User.objects.filter(email=email).exists():

            messages.info(request, 'Email taken')
            return redirect('signup')

        else:

            user = User.objects.create_user(
                username=username,
                password=password,
                email=email,
                first_name=first_name,
                last_name=last_name
            )

            messages.success(request, f"User {username} created!")
            return redirect('signin')

    return render(request, 'signup.html')


# User sign out
def signout(request):

    auth.logout(request)
    return redirect('/')
