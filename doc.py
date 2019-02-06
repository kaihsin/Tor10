import Tor10


a = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,3)])
a.SetElem([4,-3,0,\
           2,-1,2,\
           1, 5,7])
b = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,3)],is_diag=True)
b.SetElem([1,2,3])

out = Tor10.Det(a)
print(out)
out = Tor10.Det(b)
print(out)

exit(1)


bds_x = [Tor10.Bond(Tor10.BD_IN,5),Tor10.Bond(Tor10.BD_OUT,5),Tor10.Bond(Tor10.BD_OUT,3)]
x = Tor10.UniTensor(bonds=bds_x, labels=[4,3,5])
x.Print_diagram()
x.CombineBonds([4,3])
x.Print_diagram()
exit(1)


a = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,4)],labels=[1,2])
print(a.labels)

a = Tor10.Bond(Tor10.BD_IN,3)
b = Tor10.Bond(Tor10.BD_OUT,4)
c = Tor10.Bond(Tor10.BD_OUT,2,qnums=[[0,1,-1],[1,1,0]])
d = Tor10.Bond(Tor10.BD_OUT,2,qnums=[[1,0,-1],[1,0,0]]) 
e = Tor10.Bond(Tor10.BD_OUT,2,qnums=[[1,0],[1,0]])

a.combine(b)
print(a)

c.combine(d)
print(c)



