function zk = disc(x,muk,sigk,pck)

zk = (-0.5*(x-muk)*inv(sigk)*(x-muk)')-(0.5*log(det(sigk)))+log(pck);

end