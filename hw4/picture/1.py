Li=[]
def L(phi,a,b,alpha):
    l1=np.sum(phi*((x.dot((digamma(a)-digamma(a+b))))+(20-x).dot((digamma(b)-digamma(a+b)))+digamma(alpha)-digamma(np.sum(alpha))))
    l2=np.sum((a0-1)*(digamma(a)-digamma(a+b))+(b0-1)*(digamma(b)-digamma(a+b)))
    l3=np.sum((alpha0-1)*(digamma(alpha)-digamma(np.sum(alpha))))
#     e1=np.sum(-np.log(gamma(alpha)*gamma(alpha)/gamma(2*alpha)))+np.sum((alpha-1)*digamma(alpha))
#     e2=np.sum(np.log(gamma(a)*gamma(b)/gamma(a+b))-(a-1)*digamma(a)-(b-1)*digamma(b)+(a+b-2)*digamma(a+b))
    e1=np.sum(np.sum(gammaln(alpha))-gammaln(np.sum(alpha)))-np.sum((alpha-1)*digamma(alpha))+(np.sum(alpha)-K)*digamma(np.sum(alpha))
    e2=np.sum(gammaln(a)+gammaln(b)-gammaln(a+b)-(a-1)*digamma(a)-(b-1)*digamma(b)+(a+b-2)*digamma(a+b))
#     l3=0
#     e1=0
#     e2=0
    e3=-np.sum(phi*np.log(phi))
    l=l1+l2+l3+e1+e2+e3
    print("l1",l1,"l2",l2,"l3",l3,"e1",e1,"e2",e2,"e3",e3)
    Li.append(l)