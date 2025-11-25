f=open("p.txt","w")
n=int(input("Enter the value of n:"))
x=[]
for i in range(n):
        name=input("Enter name:")
        x.append(name)
f.writelines(str(x))
        
