class Person:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def say(self):
        print(self.name, self.age)  

    def ovearage(self, age):
        return self.age > age    

taro = Person("TARO", 17)
taro.say()

if taro.ovearage(20):
    print("over")
else:
    print("under")
    


pass