# Variables
name = "Ukoha"
age = 22
height = 6.1
is_student = True


#Print them
print(name)
print(age)
print(height)
print(is_student)


#Doing math variables
print(age + 8)
print(height * 2)
print("My name is " + name)

#Taking input from user
user_name = input("what is your name? ")
user_age = input("How old are you? ")
print("Hello " + user_name + " you are " + user_age +" years old")


#If statements
score = 30

if score >= 50:
    print("You passed")
else:
    print("You failed")


#Medical Ai decision
heart_rate = 40

if heart_rate > 100:
    print("WARNING: Heart rate is too high!")
elif heart_rate < 60:
    print("WARNING: Heart rate is too low!")
else:
    print("heart rate is normal")
# For loop
for i in range(5):
    print("Patient number:", i)

#looping through patient heart rates
patient_rates = [120, 75, 45, 98, 110]

for rates in patient_rates:
    if rates > 100:
        print(rates, "- Too high!")
    elif rates < 60:
        print(rates, "- Too low!")
    else:
        print(rates, "- Normal")

#Functions
def check_heart_rate(rate):
    if rate > 100:
        return "Too high!"
    elif rate < 60:
        return "Too low!"
    else:
        return "Normal"
    
print(check_heart_rate(120))
print(check_heart_rate(75))
print(check_heart_rate(40))

#Dictionary - patient record
patient = {
    "name": "Ukoha",
    "age": 22,
    "heart_rate": 75,
    "blood_pressure": 120
}


print(patient["name"])
print(patient["heart_rate"])
print(patient["blood_pressure"])

#Mini patient monitoring system
patients = [
    {"name": "John", "heart_rate": 120},
    {"name": "Ada", "heart_rate": 55},
    {"name": "Chidi", "heart_rate": 80},
]

for patient in patients:
    result = check_heart_rate(patient["heart_rate"])
    print(patient["name"], "->", result)