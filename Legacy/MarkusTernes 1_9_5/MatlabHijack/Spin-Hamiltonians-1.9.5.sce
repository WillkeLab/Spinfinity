////////////////////////////////////////////////////////////////////////////////
//
//    Spin Hamiltonians 1.9.5.sce
//
//    Stuttgart, Aachen, Jülich 14 November 2020
//
//    (C) Markus Ternes
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
////////////////////////////////////////////////////////////////////////////////

funcprot(1);
ieee(2)

////////////////////////////////////////////////////////////////////////////////
// Basic spin operators
////////////////////////////////////////////////////////////////////////////////


%muBohr=0.057883818;        //[meV/T]
%kBoltzmann=0.0861733;    //[meV/K]

function [e]=Splus(S)
  //prepare the S+ matrix
  //S.S must be the spin of the system
  e = sqrt((1:2*S.S).*(2*S.S:-1:1));
  e=diag(e,+1);
endfunction

function [e]=Sminus(S)
  //prepare the S- matrix
  //S.S must be the spin of the system
  e = sqrt((1:2*S.S).*(2*S.S:-1:1));
  e=diag(e,-1);
endfunction

function [e]=Sx(S)
  //prepare the Sx matrix
  //S.S must be the spin of the system
  e=0.5*(Splus(S)+Sminus(S));
endfunction

function [e]=Sy(S)
  //prepare the Sy matrix
  //S.S must be the spin of the system
  e=-0.5*%i*(Splus(S)-Sminus(S));
endfunction

function [e]=Sz(S)
  //prepare the Sz matrix with diagonal elements -S, -S+1,...,+S
  //S.S must be the spin of the system
  e=diag([S.S:-1:-S.S]);
endfunction

function [e]=Sarb(S,varargin)
  //prepare a S matrix with arbitary directions
  //S.S must be the spin of the system
  //varargin must contain either the rotation angles theta and phi
  //or the direction (x,y,z)
  [lhs,rhs]=argn(0);
  if rhs > 3 then
       x=varargin(1);
       y=varargin(2);
       z=varargin(3);
       theta=acos(z/sqrt(x^2+y^2+z^2));
       if isnan(theta) then
           theta=0;
       end
       phi=atan(y,x);
  else
       theta=varargin(1);
       phi=varargin(2);
  end
  e=expm(-%i*Sy(S)*theta)*Sz(S)*expm(%i*Sy(S)*theta); //rotate around y
  e=expm(-%i*Sz(S)*phi)*e*expm(%i*Sz(S)*phi);         //rotate around z
endfunction

function [e]=S1(S)
    //prepare the unity matrix for system S
    //S.S must be the spin of the system
    e=diag(ones(S.S*2+1,1));
endfunction
    
function [e]=S2(S)
 //prepare the S^2 matrix
 //S.S must be the spin of the system
  e=Sx(S)*Sx(S)+Sy(S)*Sy(S)+Sz(S)*Sz(S);
endfunction

function [e]=Srot(S,phi,n)
  // prepare the rotational matrix 
  // of the spin system S 
  // rotation angle phi
  // rotational axis n (must have length 1)
  e=expm(-%i*phi*(Sx(S)*n(1)+Sy(S)*n(2)+Sz(S)*n(3)));
endfunction

function [e]=commutator(A,B)
  //calculate the commutator of the matrices A and B
  e=A*B-B*A;
endfunction

function [e]=anticommutator(A,B)
  //calculate the anticommutator of the matrices A and B
  e=A*B+B*A;
endfunction

function [e,str]=SStev(S,a,b)
    //prepare the Stevens Operator S_a^b
    //
    //S.S must be the spin of the system
    //a=2,4,6
    //b=-a,-a+1,...,+a
    //
    //the prefacors have been taken from
    //http://www2.cpfs.mpg.de/~rotter/homepage_mcphase/manual/node132.html
    //
    x=Sx(S);
    y=Sy(S);
    z=Sz(S);
    p=Splus(S);
    m=Sminus(S);
    I=S1(S);
    ss=S.S*(S.S+1);
    select a
    case 2 then
        
        select abs(b)
        case 0 then
            e=3*z^2-ss*I;
            //str="$3\hat{S}_z^2-S(S+1)I$"
            str="$3\hat{S}_z^2$"
            
        case 1 then
            if b>0 then 
                str="$1/2 [\hat{S}_x, \hat{S}_z]_+$"
                e=0.5*anticommutator(x,z);
            else
                str="$1/2 [\hat{S}_y, \hat{S}_z]_+$"
                e=0.5*anticommutator(y,z);
            end;
            
        case 2 then
            if b>0 then
                e=0.5*(p^2+m^2)
                str="$1/2 (\hat{S}_+^2+\hat{S}_-^2)$"
            else
                e=-%i*0.5*(p^2-m^2);
                str="$-i/2 (\hat{S}_+^2-\hat{S}_-^2)$"
            end
            
        else
            e=[];str="$\rm invalid$"
        end;
        
    case 4 then
        select abs(b)
        case 0 then
            e=35*z^4-(30*ss-25)*z^2+(3*ss^2-6*ss)*I;
//          str="$35\hat{S}_z^4-(30S(S+1)-25)\hat{S_z}^2+(3S^2(S+1)^2-6S(S+1))\hat{I}$"
            str="$35\hat{S}_z^4-(30S(S+1)-25)\hat{S_z}^2$"
        case 1 then
            if b>0 then 
                pm=p+m; c=1;
                str="$1/2 [7\hat{S}_z^3-(3S(S+1)+1)\hat{S}_z, \hat{S}_++\hat{S}_-]_+$"
            else
                pm=p-m; c=-%i;
                str="$-i/2 [7\hat{S}_z^3-(3S(S+1)+1)\hat{S}_z, \hat{S}_+-\hat{S}_-]_+$"
            end;
            e=0.5*c*anticommutator(7*z^3-(3*ss+1)*z,pm);
        case 2 then
            if b>0 then 
                pm=p^2+m^2; c=1;
                str="$1/2 [7\hat{S}_z^2-(S(S+1)+5)\hat{I}, \hat{S}_+^2+\hat{S}_-^2]_+$"
            else
                pm=p^2-m^2; c=-%i;
                str="$-i/2 [7\hat{S}_z^2-(S(S+1)+5)\hat{I}, \hat{S}_+^2-\hat{S}_-^2]_+$"
            end;
            e=0.5*c*anticommutator(7*z^2-(ss+5)*I,pm);
        case 3 then
            if b>0 then 
                pm=p^3+m^3; c=1;
                str="$1/2 [\hat{S}_z, \hat{S}_+^3+\hat{S}_-^3]_+$"
            else
                pm=p^3-m^3; c=-%i;
                str="$-i/2 [\hat{S}_z, \hat{S}_+^3-\hat{S}_-^3]_+$"
            end;
            e=0.5*c*anticommutator(z,pm);
        case 4 then
            if b>0 then 
                e=0.5*(p^4+m^4);
                str="$1/2 (\hat{S}_+^4+\hat{S}_-^4)$"
            else
                e=-0.5*%i*(p^4+m^4);
                str="$-i/2 (\hat{S}_+^4-\hat{S}_-^4)$"
            end;
        else
            e=[];str="$\rm invalid$"
        end;
    case 6 then
        select abs(b)
        case 0 then
            e=231*z^6-(315*ss-735)*z^4+(105*ss^2-525*ss+294)*z^2-(5*ss^3-40*ss^2+60*ss)*I;
//            str="$231\hat{S}_z^6-(315S(S+1)-735)\hat{S}_z^4+(105S^2(S+1)^2-525S(S+1)+294)\hat{S}_z^2+(5S^3(S+1)^3-40S^2(S+1)^2+60S(S+1))\hat{I}$";
            str="$231\hat{S}_z^6-(315S(S+1)-735)\hat{S}_z^4+(105S^2(S+1)^2-525S(S+1)+294)\hat{S}_z^2$";
        case 1 then
            if b>0 then 
                pm=p+m; c=1;
                str="$1/2 [33\hat{S}_z^5-(30S(S+1)+15)\hat{S}_z^3+(5S^2(S+1)^2-10S(S+1)+12)\hat{S}_z, \hat{S}_++\hat{S}_-]_+$";
            else
                pm=p-m; c=-%i;
                str="$-i/2 [33\hat{S}_z^5-(30S(S+1)-15)\hat{S}_z^3+(5S^2(S+1)^2-10S(S+1)+12)\hat{S}_z, \hat{S}_+-\hat{S}_-]_+$";
            end;
            e=0.5*c*anticommutator(33*z^5-(30*ss-15)*z^3+(5*ss^2-10*ss+12)*z,pm);
        case 2 then
            if b>0 then 
                pm=p^2+m^2; c=1;
                str="$1/2 [33\hat{S}_z^4-(18S(S+1)+123)\hat{S}_z^2+(S^2(S+1)^2+10S(S+1)+102)\hat{I}, \hat{S}_+^2+\hat{S}_-^2]_+$"
            else
                pm=p^2-m^2; c=-%i;
                str="$-i/2 [33\hat{S}_z^4-(18S(S+1)+123)\hat{S}_z^2+(S^2(S+1)^2+10S(S+1)+102)\hat{I}, \hat{S}_+^2-\hat{S}_-^2]_+$"
            end;
            e=0.5*c*anticommutator(33*z^4-(18*ss+123)*z^2+(ss^2+10*ss+102)*I,pm);
        case 3 then
            if b>0 then 
                pm=p^3+m^3; c=1;
                str="$1/2 [11\hat{S}_z^3-(3S(S+1)+59)\hat{S}_z, \hat{S}_+^3+\hat{S}_-^3]_+$"
            else
                pm=p^3-m^3; c=-%i;
                str="$-i/2 [11\hat{S}_z^3-(3S(S+1)+59)\hat{S}_z, \hat{S}_+^3-\hat{S}_-^3]_+$"
            end;
            e=0.5*c*anticommutator(11*z^3-(3*ss+59)*z,pm);
        case 4 then
            if b>0 then 
                pm=p^4+m^4; c=1;
                str="$1/2 [11\hat{S}_z^2-(S(S+1)+38)\hat{I}, \hat{S}_+^4+\hat{S}_-^4]_+$"
            else
                pm=p^4-m^4; c=-%i;
                str="$-i/2 [11\hat{S}_z^2-(S(S+1)+38)\hat{I}, \hat{S}_+^4-\hat{S}_-^4]_+$"
            end;
            e=0.5*c*anticommutator(11*z^2-(ss+38)*I,pm);
        case 5 then
            if b>0 then 
                e=0.5*anticommutator(z,p^5+m^5);
                str="$1/2 [\hat{S}_z,\hat{S}_+^5+\hat{S}_-^5]_+$"
            else
                e=-0.5*%i*anticommutator(z,p^5-m^5);
                str="$-i/2 [\hat{S}_z,\hat{S}_+^5-\hat{S}_-^5]_+$"
            end;
        case 6 then
            if b>0 then 
                e=0.5*(p^6+m^6);
                str="$1/2 (\hat{S}_+^6+\hat{S}_-^6)$"
            else
                e=-0.5*%i*(p^6-m^6);
                str="$-i/2 (\hat{S}_+^6-\hat{S}_-^6)$"
            end;
        else
            e=[];str="$\rm invalid$"
        end;
    else
        e=[];str="$\rm out\ of\ range$"
    end;
endfunction


function [e]=dmatrix(n)
    // create the Density Matrix for spin 1/2 electrons which have a polarization
    // in the direction n and a polarization strength -1<|n|<1.
    S.S=1/2;
    e=(1/2*S1(S)+n(1)*Sx(S)+n(2)*Sy(S)+n(3)*Sz(S));
endfunction

////////////////////////////////////////////////////////////////////////////////
// Basic spin operators for spin ensembles
////////////////////////////////////////////////////////////////////////////////

function [e]=Sxn(t)
  //prepare the sum of all Sx matrices for an ensemble t of spins
  //Szn([Fe,Co,...])
  b=size(t,2);     // determine the number of coupled spins
  w=1;
  for i=1:b,
      w=w*(2*t(i).S+1),
  end
  e=spzeros(w,w);
  
  for i=1:b,
      x=sparse(1);
      for j=1:i-1, 
          x=x.*.sparse(S1(t(j))); 
      end;
      x=x.*.sparse(Sx(t(i)));
     for j=i+1:b, 
          x=x.*.sparse(S1(t(j)));
     end;
     e=e+x;
  end;
endfunction

function [e]=Syn(t)
  //prepare the sum of all Sy matrices for an ensemble t of spins
  //Szn([Fe,Co,...])
  b=size(t,2);     // determine the number of coupled spins

  w=1;
  for i=1:b,
      w=w*(2*t(i).S+1),
  end
  e=spzeros(w,w);
  
  for i=1:b,
      x=sparse(1);
      for j=1:i-1, 
          x=x.*.sparse(S1(t(j))); 
      end;
      x=x.*.sparse(Sy(t(i)));
     for j=i+1:b, 
          x=x.*.sparse(S1(t(j)));
     end;
     e=e+x;
  end;
endfunction

function [e]=Szn(t)
  //prepare the sum of all Sz matrices for an ensemble t of spins
  //Szn([Fe,Co,...])
  b=size(t,2);     // determine the number of coupled spins

  w=1;
  for i=1:b,
      w=w*(2*t(i).S+1),
  end
  e=spzeros(w,w);
  
  for i=1:b,
      x=sparse(1);
      for j=1:i-1, 
          x=x.*.sparse(S1(t(j))); 
      end;
      x=x.*.sparse(Sz(t(i)));
     for j=i+1:b, 
          x=x.*.sparse(S1(t(j)));
     end;
     e=e+x;
  end;
endfunction

function [e]=Sarbn(t,varargin)
  //prepare a sum of all S matrices arbitary direction
  //for an ensemble t of spins
  //varargin must contain either the rotation angles theta and phi
  //or the direction (x,y,z)
  //Sarbn([Fe,Co,...],x,y,z)
  b=size(t,2);     // determine the number of coupled spins
  
  [lhs,rhs]=argn(0);
  if rhs > 3 then
       x=varargin(1);
       y=varargin(2);
       z=varargin(3);
       theta=acos(z/sqrt(x^2+y^2+z^2));
       if isnan(theta) then
           theta=0;
       end;
       phi=atan(y,x);
  else
       theta=varargin(1);
       phi=varargin(2);
  end
 
  w=1;
  for i=1:b,
      w=w*(2*t(i).S+1),
  end
  e=spzeros(w,w);
  
  for i=1:b,
      x=sparse(1);
      for j=1:i-1, 
          x=x.*.sparse(S1(t(j))); 
      end;
      y=expm(-%i*Sy(t(i))*theta)*Sz(t(i))*expm(%i*Sy(t(i))*theta); //rotate around y
      y=expm(-%i*Sz(t(i))*phi)*y*expm(%i*Sz(t(i))*phi);      //rotate around z
      x=x.*.sparse(y);
     for j=i+1:b, 
          x=x.*.sparse(S1(t(j)));
     end;
     e=e+x;
  end;
endfunction

function [e]=SStevn(t,a,b)
  //prepare the sum of all Stevens matrices O_a^b for an ensemble t of spins
  //SStevn([Fe,Co,...])
  b=size(t,2);     // determine the number of coupled spins

  w=1;
  for i=1:b,
      w=w*(2*t(i).S+1),
  end
  e=spzeros(w,w);
  
  for i=1:b,
      x=sparse(1);
      for j=1:i-1, 
          x=x.*.sparse(S1(t(j))); 
      end;
      x=x.*.sparse(SStev(t(i),a,b));
     for j=i+1:b, 
          x=x.*.sparse(S1(t(j)));
     end;
     e=e+x;
  end;
endfunction

function [e]=S2n(t)
  //prepare the S^2 matrix for an ensemble t of spins
  //S2n([Fe,Co,...])
  e=Sxn(t)^2+Syn(t)^2+Szn(t)^2;
endfunction

function [e]=Srotn(S,phi,n)
  // prepare the rotational matrix 
  // of the spin system S 
  // rotation angle phi
  // rotational axis n (must have length 1)

  e=expm(full(-%i*phi*(Sxn(S)*n(1)+Syn(S)*n(2)+Szn(S)*n(3))));      

endfunction


function [dm]=hermitianize(dm)
    dm=dm-%i*diag(imag(diag(dm))) //remove spourious imaginary parts on the diagonal
    //diagonal must be one
//    dm=dm./sum(trace(dm));
    //test of Hermitianity
    if ~and(dm'==dm) then
        //make it Hermitian!
    dm=(dm+dm')/2;
    end
endfunction

////////////////////////////////////////////////////////////////////////////////
// Basic Energy Hamiltonians
////////////////////////////////////////////////////////////////////////////////

function [e]=HZ(S,B)
    //prepare Zeemann Hamiltonian HZ
    //S is the spin system with at least
    //S.S as the spin
    //S.g as the gyomagnetic factor
    //B is the external field 
    a=length(B); //check if B is 1dim or 3dim
    if a<>1 & a<>3 then error(5);abort;
    end;
    if a==1 then B=[0,0,B]; //assign B to B_Z if B is 1dim
    end;
    //check it g is 1-dim or 3-dim //new in 1.9.4!
    a=length(S.g);
    if a<>1 & a<>3 then error(5);abort;
    end;
    if a==1 then S.g=[S.g,S.g,S.g]; //assign g to all directions if g is 1dim
    end;
    if exists("%renormalize_E") then
        if %renormalize_E then
            if length(S.J)==1 then
                S.g=S.g*(1-S.J)
            else
                S.g=S.g.*(1-S.J); 
            end
            //The renormalization follows 
            //E.L. Wolf and D.L. Losee Phys. Lett. A 29, 334 (1969)
       end,
    end,
    e=-%muBohr*(S.g(1)*B(1)*Sx(S)+S.g(2)*B(2)*Sy(S)+S.g(3)*B(3)*Sz(S))
    // the minus is important because it forces the spins to be alligned with the 
    // field direction.
endfunction

function [e]=Haniso(S)
    //prepare the anisotropy Hamiltonian for the spin system S.
    //The spin system contains at least
    //S.S as the spin
    //S.D, S.E, S.DD, and S.G will be used for the calculation if they exist
    //OR
    //if S.use_Stevens exists and is true than the values in S.Stevens 
    //will be used
    //if S.alpha, S.beta, and S.gamma exists
    //then the system will be rotated successively around the x, y, and z axes
    
    txt=getfield(1,S);
    //check what parameters we have
    if grep(txt,'use_Stevens')==[] then
        use_st=%f;
    else
        use_st=S.use_Stevens
    end
    
    e=0*S1(S);  //initialize matrix 
    if use_st==%f | use_st==0 then
        //use standard anisotropy parameters to calculate Haniso
        if grep(txt,'D')==[] then 
            S.D=0, 
        end;
        if grep(txt,'E')==[] then 
            S.E=0, 
        end;
        e=S.D*(Sz(S)*Sz(S))+S.E*(Sx(S)*Sx(S)-Sy(S)*Sy(S)),
        if grep(txt,'DD') then 
            e=e+S.DD*(Sz(S)^4),
        end;
        if grep(txt,'G') then 
            e=e+S.G*(Splus(S)^4+Sminus(S)^4),
        end;
    else
        //use Stevensoperator to calculate Haniso
        [az,bz]=size(S.Stevens);
        for a=2:2:2*az,
            for b=0:1:bz-1,
                ee=SStev(S,a,b);
                if ee<>[] then
                    e=e+S.Stevens(a/2,b+1)*ee,
                end;
            end;
        end;
    end;
    
    if grep(txt,'alpha') then
        e=expm(-%i*Sx(S)*S.alpha)*e*expm(%i*Sx(S)*S.alpha); //rotate around x
        e=expm(-%i*Sy(S)*S.beta)*e*expm(%i*Sy(S)*S.beta);   //rotate around y
        e=expm(-%i*Sz(S)*S.gamma)*e*expm(%i*Sz(S)*S.gamma); //rotate around z
    end

endfunction

function [e]=HHeisenberg(t,J,n,m)
    //Create a Hamiltonian which couple two spins (n and m) 
    //from an ensemble t of spins with the Heisenberg interaction J.
    //HHeisenberg([Fe,Co,...],J,n,m)
    
    a=length(J); //check if J is 1dim or 3dim
    if a<>1 & a<>3 then error(5);abort;
    end;
    if a==1 then J=[J,J,J]; //if J is 1dim assign Jx=J,Jy=J,Jz=J
    end;

    b=size(t,2)     // determine the number of coupled spins
    //make some checks!
    if (b<n) | (b<m) | (n==m) then error(1);abort;
    end
    if n>m then p=n;n=m;m=p; //swap n and m
    end

    x=sparse(1);y=x;z=x;
    for j=1:n-1, 
        m1=sparse(S1(t(j)));
        x=x.*.m1; 
        y=y.*.m1; 
        z=z.*.m1; 
    end;
    x=x.*.sparse(Sx(t(n)));
    y=y.*.sparse(Sy(t(n)));
    z=z.*.sparse(Sz(t(n)));
    for j=n+1:m-1, 
        m1=sparse(S1(t(j)));
        x=x.*.m1; 
        y=y.*.m1; 
        z=z.*.m1; 
    end;
    x=x.*.sparse(Sx(t(m)));
    y=y.*.sparse(Sy(t(m)));
    z=z.*.sparse(Sz(t(m)));
    for j=m+1:b, 
        m1=sparse(S1(t(j)));
        x=x.*.m1; 
        y=y.*.m1; 
        z=z.*.m1; 
    end;

    e=J(1)*x+J(2)*y+J(3)*z;
endfunction

function [e]=HDM(t,D,n,m)
    //Create a Hamiltonian which couples two spins (n and m) 
    //from an ensemble t of spins with the DM interaction D.
    //HDM([Fe,Co,...],D,n,m) 
     
    b=size(t,2);     // determine the number of coupled spins
    //make some checks!
    if (b<n) | (b<m) | (n==m) then error(1);abort;
    end
    if n>m then p=n;n=m;m=p;D=-D; //swap n and m and change sign of D
    end
    
    x=sparse(1);y=x;z=x;xx=x;yy=x;zz=x;
    
    for j=1:n-1, 
        x=x.*.sparse(S1(t(j)));
        xx=x;y=x;yy=x;z=x;zz=x;
    end;
    x=x.*.sparse(Sy(t(n)));xx=xx.*.sparse(Sz(t(n)));
    y=y.*.sparse(Sz(t(n)));yy=yy.*.sparse(Sx(t(n)));
    z=z.*.sparse(Sx(t(n)));zz=zz.*.sparse(Sy(t(n)));
    for j=n+1:m-1, 
        m1=sparse(S1(t(j)));
        x=x.*.m1;xx=xx.*.m1; 
        y=y.*.m1;yy=yy.*.m1; 
        z=z.*.m1;zz=zz.*.m1;  
    end;
    x=x.*.sparse(Sz(t(m)));xx=xx.*.sparse(Sy(t(m)));
    y=y.*.sparse(Sx(t(m)));yy=yy.*.sparse(Sz(t(m)));
    z=z.*.sparse(Sy(t(m)));zz=zz.*.sparse(Sx(t(m)));
    for j=m+1:b, 
        m1=sparse(S1(t(j)));
        x=x.*.m1;xx=xx.*.m1; 
        y=y.*.m1;yy=yy.*.m1; 
        z=z.*.m1;zz=zz.*.m1;  
    end;
    e=D(1)*(x-xx)+D(2)*(y-yy)+D(3)*(z-zz);
endfunction

function [e]=Hdd(t,r,n,m)
    
    //Create a Hamiltonian which couples two spins (n and m) 
    //from an ensemble t of spins with the dipole-dipole interaction.
    //the vector r points thereby from spin n to spin m in the actual 
    //coordinate system
    //Hdd([Fe,Co,...],r,n,m) 
    
    //check if r is 3dim
    if length(r)<>3 then 
        error(5);
        abort;
    end;
    
    b=size(t,2);     // determine the number of coupled spins
    //make some checks!
    if (b<n) | (b<m) | (n==m) then error(1);abort;
    end
    if n>m then p=n;n=m;m=p;r=-r; //swap n and m and change sign of r
    end
    
    ra=sqrt(sum(r.^2));
    if ra==0 then error(1);abort;
    end
    
    x=sparse(1);y=x;z=x;xx=x;yy=y;zz=z;     //initialize matrices
    
    for j=1:n-1, 
        x=x.*.sparse(S1(t(j)));
        xx=x;y=x;yy=x;z=x;zz=x;
    end;
    x=x.*.sparse(Sx(t(n)));xx=xx.*.sparse(S1(t(n)));
    y=y.*.sparse(Sy(t(n)));yy=yy.*.sparse(S1(t(n)));
    z=z.*.sparse(Sz(t(n)));zz=zz.*.sparse(S1(t(n)));
    for j=n+1:m-1, 
        m1=sparse(S1(t(j)));
        x=x.*.m1;xx=xx.*.m1; 
        y=y.*.m1;yy=yy.*.m1; 
        z=z.*.m1;zz=zz.*.m1;  
    end;
    x=x.*.sparse(S1(t(m)));xx=xx.*.sparse(Sx(t(m)));
    y=y.*.sparse(S1(t(m)));yy=yy.*.sparse(Sy(t(m)));
    z=z.*.sparse(S1(t(m)));zz=zz.*.sparse(Sz(t(m)));
    for j=m+1:b, 
        m1=sparse(S1(t(j)));
        x=x.*.m1;xx=xx.*.m1; 
        y=y.*.m1;yy=yy.*.m1; 
        z=z.*.m1;zz=zz.*.m1;  
    end;
    
    rx=r(1)/ra; ry=r(2)/ra; rz=r(3)/ra;
   
    e=x*xx*(3*rx^2-1)+y*yy*(3*ry^2-1)+z*zz*(3*rz^2-1)+...
    anticommutator(x,y)*(3*rx*ry)+...
    anticommutator(x,z)*(3*rx*rz)+...
    anticommutator(y,z)*(3*ry*rz);
    
    e=e/(ra^3);
    //the "magic" angle would be reached for [0,0.816,0.578]

endfunction



function [e]=HAni(t,B)
    //Create a Hamiltonian in which all spins of the ensemble t
    //have their anisotropy and Zeeman energy included
    //HAni([Fe,Co,...],B)  
    b=size(t,2)     // determine the number of coupled spins
    if b==1  then   //only one atom
        e=Haniso(t)+HZ(t,B);
    else
        w=1;
        for i=1:b,
            w=w*(2*t(i).S+1),
        end
        e=spzeros(w,w);

        for i=1:b,
            x=sparse(1);
            for j=1:i-1, 
                x=x.*.sparse(S1(t(j))); 
            end;
            x=x.*.sparse(Haniso(t(i))+HZ(t(i),B))
            for j=i+1:b, 
                x=x.*.sparse(S1(t(j)));
            end;
            e=e+x;
        end;    
    end
endfunction

function H=hamiltonian(experiment)
    //This function calculates the total Hamiltonian of a coupled spin system
    //
    // experiment has to contain at least the following information:
    //
    // experiment.atom                  (either the parameters of an atom or a list of atoms)
    // experiment.B                     (the external field)
    //
    // furthermore, the following additional parameters are possible
    // experiment.heisenberg_coupling   (the global coupling constant)
    // experiment.matrix                (the Heisenberg coupling constant between the (i,i+1)th atom)
    // experiment.matrixDM              (the DM coupling vector between the (i,i+1)th atom)
    
    //build the Zeeman and Anisotropy Hamiltonian
     
    H=HAni(experiment.atom,experiment.B);
    
    if size(experiment.atom,2)>1  then
        
        //check if we have Heisenberg coupling
        if or(getfield(1,experiment)=='matrix') then
            M=experiment.matrix;
            J=experiment.heisenberg_coupling;
            if J<>0 then
               for i=1:size(M,1),
                   for j=2:size(M,2)+1,
                       if j>i then
//                          H=Hdd(experiment.atom,evstr(M(i,j-1))*J,i,j)+H; 
                          H=HHeisenberg(experiment.atom,evstr(M(i,j-1))*J,i,j)+H;
                       end
                   end
               end
            end
        end
        
        //check if we have DM coupling
        if or(getfield(1,experiment)=='matrixDM') then
            M=experiment.matrixDM;
            J=experiment.heisenberg_coupling;
            if J<>0 then
               for i=1:size(M,1),
                   for j=2:size(M,2)+1,
                       if j>i then
                          H=HDM(experiment.atom,evstr(M(i,j-1))*J,i,j)+H;
                       end
                   end
               end
            end
        end
   end
   
endfunction

function eigenvalues(experiment,varargin)
    //the function calculates the eigenvectors and eigenvalues
    //and stores them in experiment.Eigenvec and experiment.Eigenval
    //
    // experiment has to contain at least the following information:
    //
    // experiment.atom                  (either the parameters of an atom or a list of atoms)
    // experiment.B                     (the external field)
    // experiment.max_no_eigenstates    (the maximal number of calculated eigenvectors)
    //
    // furthermore, the following additional parameters are possible
    // experiment.heisenberg_coupling   (the global coupling constant)
    // experiment.matrix                (the Heisenberg coupling constant between the (i,i+1)th atom)
    // experiment.matrixDM              (the DM coupling vector between the (i,i+1)th atom)
    
    // if varargin is set to <>1 no progression bar is shown (silent mode)
    
    global experiment;
    
    [lhs,rhs]=argn(0); 
    if rhs > 1 then
        v=varargin();
        v=v(1);
    else
        v=1;
    end
    
    
    H=hamiltonian(experiment); //construct Hamiltonian
    
    if v==1 then winId=progressionbar('Diagonalizing Hamiltonain. Please be patient!');
    end

       // solve the eigenvalue-problem
    if or(getfield(1,experiment)=='max_no_eigenstates') then 
        //check if max_no_eigenstates exist
        if size(H,1) < experiment.max_no_eigenstates+2 then
            H=hermitianize(H);
            [experiment.Eigenvec,experiment.Eigenval]=spec(full(H));
        else
            ofst=0;ofstold=0;
            while ofst >= 0
                H=H-ofst*speye(H); 
                [experiment.Eigenval,experiment.Eigenvec]=..
                eigs(H,speye(H),experiment.max_no_eigenstates);
                ofstold=ofstold+ofst;
                ofst=max(real(diag(experiment.Eigenval)));
            end
            if ofstold > 0 then
               experiment.Eigenval=..
               experiment.Eigenval+ofstold*eye(experiment.Eigenval);
            end
        end
    else
        [experiment.Eigenvec,experiment.Eigenval]=spec(full(H));
    end
    
    experiment.Eigenval=diag(experiment.Eigenval); //reduce Eigenval to a vector
    
    //check if there are some complex components in the Eigenvalues due to numerical accuracy
    
    if ~isreal(experiment.Eigenval) then          
        //remove imag part    
        experiment.Eigenval=real(experiment.Eigenval);
        //resort eigenvalues
        [experiment.Eigenval,k]=gsort(experiment.Eigenval,'g','i');
    
        if ~and(k==gsort(k,'g','i')) then
            experiment.Eigenvec=experiment.Eigenvec(:,k); //resort Eigenvectors
        end;
    
    end

    if v==1 then close(winId);
    end

endfunction

////////////////////////////////////////////////////////////////////////////////
// Additional Hamiltonians and matrix functions for special purposes
////////////////////////////////////////////////////////////////////////////////

function [e]=traceout(rho,t,n)
   
   // fuction to traceout parts of the full density matrix
   //    
   // rho is the density matrix of the full system
   // t is a struct array of the corresponding coupled atoms
   // n is a vector containing the number of spins in the spin list 
   // of the reduced density matrix
   // n must be a continous list i.e. [2,3,4] but NOT [2,4,5]!
   // e is finaly the reduced density matrix
   // ATTENTION! 
   // The list n contains the remaining atoms in the new density matrix
   
    n=gsort(n,'g','i');
    b=size(t,2);     // determine the number of coupled spins
    c0=1;
    for i=n,
        c0=c0*(2*t(i).S+1);
    end
    c0=zeros(c0,c0);
    e=zeros(c0);
    for i=1:length(e),
        x=sparse(1);
        for j=1:min(n)-1, 
            x=x.*.sparse(S1(t(j))); 
        end;
        c=c0; 
        c(i)=1;
        x=x.*.c;
        for j=max(n)+1:b, 
            x=x.*.sparse(S1(t(j)));
        end;
        e(i)=sum(x.*rho);
    end;
endfunction

function [e]=ptranspose(rho,t,n)
   
   // Function to partly transpose a density matrix
   // 
   // rho is the (density) matrix of the full system
   // t is a struct array of the corresponding coupled atoms
   // n is a vector containing the number of spins which shall be transposed 
   // e is finaly partially transposed (density) matrix
   
    
    zout=size(rho,2);
    b=size(t,2);     // determine the number of coupled spins
    
    for i=1:b,
        sz(i)=2*t.S(i)+1; 
    end
    
    p=[1:2*b];
    p([b+1-n,2*b+1-n]) = p([2*b+1-n,b+1-n]);
    
    e=matrix(permute(matrix(rho,[sz($:-1:1),sz($:-1:1)]),p),zout,zout);
   
endfunction

function [e]=permuterho(rho,t,n)
   
   // function to permute the submatrices of the total density matrix
    
   // rho is the (density) matrix or a state vector of the system
   // t is a struct array of the corresponding coupled atoms
   // n is a vector containing the new order of spins 
   // e is finaly the permuted (density) matrix
   
    
    zout=size(rho); //determine the size of the input
    b=size(t,2);     // determine the number of coupled spins
    
    for i=1:b,
        sz(i)=2*t.S(i)+1; //detemine the sizes of the subsystems
    end
    
    if or(zout==1) then  //state vector
        p = b+1-n($:-1:1);
        e=matrix(permute(matrix(rho,sz($:-1:1)),p),zout);
    else
         p = b+1-n($:-1:1);
         p = [p,b+p];
          e=matrix(permute(matrix(rho,[sz($:-1:1),sz($:-1:1)]),p),zout);
       
    end
   
endfunction

////////////////////////////////////////////////////////////////////////////////
// Calculations of Transition Matrix elements
////////////////////////////////////////////////////////////////////////////////

function [x,y,z,u]=etransport(experiment)
// this function calculates the transition matrix elements for the tunneling and
// interacting S=1/2 electrons in their spin base. Thus, it gives the transport 
// probabilities for arbitrary polarized electrons in tip and sample.
// experiment.ptip and experiment.psample must be the 3dim spin polarization of
// tip and sample, respectively.
// 'x' the matrix <f|sigma_x|i>
// 'y' the matrix <f|sigma_y|i>
// 'z' the matrix <f|sigma_z|i>
// 'U' the matrix <f|i> 

  e.S=1/2;
  dt=dmatrix(experiment.ptip);
  ds=dmatrix(experiment.psample);
  [wt,pt]=spec(dt);
  [ws,ps]=spec(ds);
  
  pt=diag(pt);
  ps=diag(ps);

  pp=sqrt(ps*pt');  
  
  //the factor 2 originates from the differences 
  //between Pauli-matices and S matrices
  x=(ws'*(2*Sx(e))*wt).*pp;
  y=(ws'*(2*Sy(e))*wt).*pp;
  z=(ws'*(2*Sz(e))*wt).*pp;
  u=2*(ws'*wt).*pp;
      
endfunction

function [xx]=e2transport(experiment)
// this function calculates the transition matrix elements for the tunneling and
// interacting S=1/2 electrons in their spin base when tunneling trough a 
// spectroscopically active tip, i.e. interacting with TWO spin SYSTEMS.

  nrstates=size(experiment.ptipK,1);
  
  xx=zeros(2,2,nrstates,nrstates); //prepare matrices
//  yy=zeros(2,2,nrstates,nrstates);
//  zz=zeros(2,2,nrstates,nrstates);
//  uu=zeros(2,2,nrstates,nrstates);

  if experiment.sef==0 then
    //go the easy way;
    //"source entanglement factor" ist zero, therefore we only use the 
    //ptip and psample.
    
    e.S=1/2;
    dt=dmatrix(experiment.ptip);
    ds=dmatrix(experiment.psample);
    [wt,pt]=spec(dt);
    [ws,ps]=spec(ds);
  
    pt=diag(pt);
    ps=diag(ps);

    pp=sqrt(ps*pt');  
  
    x1=2*(ws'*wt).*pp; 

    for i=1:nrstates,
        for j=1:nrstates,
            xx(:,:,i,j)=x1;
        end
    end
    
  else
     //calculate the vector bases and intensities for all possible states
     
    e.S=1/2;
    pt=zeros(2,nrstates);
    ps=zeros(2,nrstates);
    wt=zeros(2,2,nrstates);
    ws=zeros(2,2,nrstates);

    for i=1:nrstates,
      dt=dmatrix(experiment.ptipK(i,:));
      ds=dmatrix(experiment.psampleK(i,:));
      [wt(:,:,i),t]=spec(dt);
      [ws(:,:,i),s]=spec(ds);
      pt(:,i)=diag(t);
      ps(:,i)=diag(s);
    end
  
    for i=1:nrstates,
        for j=1:nrstates,
            pp=sqrt(ps(:,j)*pt(:,i)');
            //the factor 2 originates from the differences 
            //between Pauli-matices and S matrices
            xx(:,:,i,j)=2*(ws(:,:,j)'*wt(:,:,i)).*pp;
        end;
    end;
  end;


endfunction

function [x,y,z,u]=etransport_func_tip(experiment)
// this function calculates the transition matrix elements for the tunneling and
// interacting S=1/2 electrons in their spin base. Thus, it gives the transport 
// probabilities for arbitrary polarized electrons in tip and sample.
// experiment.ptip and experiment.psample must be the 3dim spin polarization of
// tip and sample, respectively.
// 'x' the matrix <f|sigma_x|i>
// 'y' the matrix <f|sigma_y|i>
// 'z' the matrix <f|sigma_z|i>
// 'U' the matrix <f|i> 
   
  nrstates=size(experiment.ptipK,1);
  
  x=zeros(2,2,nrstates,nrstates); //prepare matrices
  y=zeros(2,2,nrstates,nrstates);
  z=zeros(2,2,nrstates,nrstates);
  u=zeros(2,2,nrstates,nrstates);

  if experiment.sef==0 then
    //go the easy way; 
    [x1,y1,z1,u1]=etransport(experiment);
    for i=1:nrstates,
        for j=1:nrstates,
            x(:,:,i,j)=x1;
            y(:,:,i,j)=y1;
            z(:,:,i,j)=z1;
            u(:,:,i,j)=u1;
        end
    end
    
  else
     
    e.S=1/2;
    pt=zeros(2,nrstates);
    ps=zeros(2,nrstates);
    wt=zeros(2,2,nrstates);
    ws=zeros(2,2,nrstates);

    for i=1:nrstates,
      dt=dmatrix(experiment.ptipK(i,:));
      ds=dmatrix(experiment.psampleK(i,:));
      [wt(:,:,i),t]=spec(dt);
      [ws(:,:,i),s]=spec(ds);
      pt(:,i)=diag(t);
      ps(:,i)=diag(s);
    end
  
    for i=1:nrstates,
        for j=1:nrstates,
            pp=sqrt(ps(:,j)*pt(:,i)');
            //the factor 2 originates from the differences 
            //between Pauli-matices and S matrices
            x(:,:,i,j)=(ws(:,:,j)'*(2*Sx(e))*wt(:,:,i)).*pp;
            y(:,:,i,j)=(ws(:,:,j)'*(2*Sy(e))*wt(:,:,i)).*pp;
            z(:,:,i,j)=(ws(:,:,j)'*(2*Sz(e))*wt(:,:,i)).*pp;
            u(:,:,i,j)=2*(ws(:,:,j)'*wt(:,:,i)).*pp;
        end;
    end;
  end;
     
endfunction

function [e, er, en, enr]=e_epsilon(experiment)
    // calculate the tunneling rate in 3rd order
    // via all possible tunneling pathes FOR THE ELECTRONS
    // output are 4 tensors (size 4x4x4) 
    // which give the probabiliy to tunnel from i->j->k
    // e is the 'normal' tensor for tunneling from tip to sample
    // er is the 'time-reversed' tensor for tunneling from tip to sample     
    // en is the 'normal' tensor for tunneling from sample to tip
    // enr is the 'time-reversed' tensor for tunneling from sample to tip
    // experiment has to contain:
    // 'experiment.ptip':     spin polarization of the tip
    // 'experiment.psample':  spin polarization of the sample
   
    [x,y,z,u]=etransport(experiment);
    
    tempexperiment=experiment;
    tempexperiment.ptip=tempexperiment.psample;
    [xs,ys,zs,us]=etransport(tempexperiment);
    
    mone=ones(1,2);   
    so1=list(x, y, z, u);
    so2=list(xs, ys, zs, us);
    so3=list(x', y', z', u');
    e=zeros(4,4,4);er=e;en=e;enr=e;
    for j=1:4,
       for k=1:4,
          for l=1:4,
             for in=1:2, 
                // the following line calculates 
                // <in|so3(l)|b><b|so2(k)|a><a|so1(j)|in>
                // with |in> as one of the two possible states of the initial 
                // electron state and |a>, |b> arbitraly intermediate states.
                // Therefore 'temp' is a 2x2 matrix which we have to sum up!
                temp=(mone'.*.so3(l)(in,:)).*(so2(k)).'.*(mone.*.so1(j)(:,in));    
                e(j,k,l)=e(j,k,l)+sum(temp);
                
                temp=-(mone'.*.so3(l)(in,:)).*(mone.*.so1(k)(:,in)).*(so2(j).');
                er(j,k,l)=er(j,k,l)+sum(temp);
                
                temp=-(mone.*.so1(l)(:,in)).*(so2(k)).'.*(mone'.*.so3(j)(in,:));
                en(j,k,l)=en(j,k,l)+sum(temp);
                
                temp=(mone.*.so1(l)(:,in)).*(mone'.*.so3(k)(in,:)).*(so2(j).');
                enr(j,k,l)=enr(j,k,l)+sum(temp);
             end;
          end;
       end;
    end;
endfunction

function [out1,out2,out3,out4]=Rate2nd(experiment)
  // calculate the tunneling matrix elements up to 2nd order
  // via all possible tunneling pathes
  // i.e.
  // out1= <f,u'|Sx+Sy+Sz+U|i,u>
  // out2= <f,u'|Sx+Sy+Sz+U|i,d>
  // out3= <f,d'|Sx+Sy+Sz+U|i,u>
  // out4= <f,d'|Sx+Sy+Sz+U|i,d>
  // with u/d and u'/d' as the two eigenstates in tip and sample, respectively.
  // experiment has to contain:
  // 'experiment.ptip':     spin polarization of the tip
  // 'experiment.psample':  spin polarization of the sample
  // 'experiment.Eigenvec': matrix of the eigenvectors of the spin ensemble
  // 'experiment.atom':     a list with the atoms of the system 
  // 'experiment.position': position in the atom list at which the rate is calc.
  
  //first we calculate the transition matrices for the tunneling electron
  [x,y,z,u]=etransport(experiment);

  E=experiment.Eigenvec;
  if ~isvector(experiment.atom) then
      // go the easy way! It's only one spin-system
      S=experiment.atom;

      matrxx=Sx(S);
      matrxy=Sy(S);
      matrxz=Sz(S);
      matrxu=S1(S);
      
  else
      nr=experiment.position;
      natoms=size(experiment.atom);
      matrxx=sparse(1);

      for i=1:(nr-1),
          mm=sparse(S1(experiment.atom(i)));
          matrxx=matrxx.*.mm;
      end
      matrxy=matrxx;
      matrxz=matrxx;
      matrxu=matrxx;
      S=experiment.atom(nr);
      matrxx=matrxx.*.Sx(S);
      matrxy=matrxy.*.Sy(S);
      matrxz=matrxz.*.Sz(S);
      matrxu=matrxu.*.S1(S);
    
      for i=nr+1:natoms(2),
          mm=sparse(S1(experiment.atom(i)));
          matrxx=matrxx.*.mm;
          matrxy=matrxy.*.mm;
          matrxz=matrxz.*.mm;
          matrxu=matrxu.*.mm;                              
      end
      
  end
  
  if length(S.J)==1 then
      S.J=[S.J ,S.J, S.J];      //new (version 1.6)
  end
 
  matrxx=S.J(1)*E'*matrxx*E;   //the multiplication of J is new (version 1.6)
  matrxy=S.J(2)*E'*matrxy*E;
  matrxz=S.J(3)*E'*matrxz*E;
  matrxu=S.U*E'*matrxu*E;
 
      out1=(matrxx*x(1)+matrxy*y(1)+matrxz*z(1)+matrxu*u(1));
      out2=(matrxx*x(2)+matrxy*y(2)+matrxz*z(2)+matrxu*u(2));
      out3=(matrxx*x(3)+matrxy*y(3)+matrxz*z(3)+matrxu*u(3));
      out4=(matrxx*x(4)+matrxy*y(4)+matrxz*z(4)+matrxu*u(4));
      
endfunction

function [out]=Rate2nd2(experiment)
  // calculate the tunneling rate up to 2nd order
  // via all possible tunneling pathes
  // i.e.
  // <f|Sx+Sy+Sz+U|i>^2             if experiment.use_T_ts==%F or nonexisitent
  //
  // <f|Tx*Sx+Ty*Sy+Tz*Sz+U|i>^2    if experiment.use_T_ts==%T
  
  // 'out' is a matrix which gives the probabiliy to tunnel from i->j
  // experiment has to contain:
  // 'experiment.ptip':     spin polarization of the tip
  // 'experiment.psample':  spin polarization of the sample
  // 'experiment.Eigenvec': matrix of the eigenvectors of the spin ensemble
  // 'experiment.atom':     a list with the atoms of the system 
  // 'experiment.position': position in the atom list at which the rate is calc.
  
  //first we calculate the transition matrices for the tunneling electron
  
  txt=getfield(1,experiment);
  if grep(txt,'use_T_ts')==[] then
      use_T_ts=%F
  else
      use_T_ts=experiment.use_T_ts;
  end
  
  [x,y,z,u]=etransport(experiment);
  
  E=experiment.Eigenvec;
  if ~isvector(experiment.atom) then
      // go the easy way! It's only one spin-system
      S=experiment.atom;

      matrxx=Sx(S);          
      matrxy=Sy(S);
      matrxz=Sz(S);
      matrxu=S1(S);
      
  else
      nr=experiment.position;
      natoms=size(experiment.atom);
      matrxx=sparse(1);

      for i=1:(nr-1),
          mm=sparse(S1(experiment.atom(i)));
          matrxx=matrxx.*.mm;
      end
      matrxy=matrxx;
      matrxz=matrxx;
      matrxu=matrxx;
      S=experiment.atom(nr);

      matrxx=matrxx.*.Sx(S);
      matrxy=matrxy.*.Sy(S);
      matrxz=matrxz.*.Sz(S);
      matrxu=matrxu.*.S1(S);

      for i=nr+1:natoms(2),
          mm=sparse(S1(experiment.atom(i)));
          matrxx=matrxx.*.mm;
          matrxy=matrxy.*.mm;
          matrxz=matrxz.*.mm;
          matrxu=matrxu.*.mm;                              
      end
  end

  if use_T_ts then
      J=experiment.T_ts;      //new (version 1.6)
  else
      J=[1 1 1];
  end

  matrxx=J(1)*E'*matrxx*E;   //the multiplication of J is new (version 1.6)
  matrxy=J(2)*E'*matrxy*E;
  matrxz=J(3)*E'*matrxz*E;
  matrxu=S.U*E'*matrxu*E;

  out=0;
  for i=1:4,
      out=out+abs(matrxx*x(i)+matrxy*y(i)+matrxz*z(i)+matrxu*u(i)).^2;
  end

endfunction

function [e,er,en,enr]=M35(experiment,in)
    // calculate the tunneling rate in 3rd order
    // via all possible tunneling pathes
    // output are 4 matrices which give the probabiliy to tunnel from in->i->j
    // e is the 'normal' matrix for tunneling from tip to sample
    // er is the 'time-reversed' matrix for tunneling from tip to sample     
    // en is the 'normal' matrix for tunneling from sample to tip
    // enr is the 'time-reversed' matrix for tunneling from sample to tip
    // experiment has to contain:
    // 'experiment.ptip':     spin polarization of the tip
    // 'experiment.psample':  spin polarization of the sample
    // 'experiment.Eigenvec': matrix of the eigenvectors of the spin ensemble
    // 'experiment.atom':     a list containing the atoms of the system 
    // 'experiment.position': is the number of the atom in the ensemble with 
    //                        which the tunneling electron interacts
    // 'experiment.jposition':is the number of the atom in the ensemble with
    //                        which the sample-to-sample electron interacts
    // 'in' is the number of the initial state

    //first we calculate the transition matrices for the electrons in 3rd order
    
   E=experiment.Eigenvec; a=size(E);
   
   txt=getfield(1,experiment);
   if grep(txt,'entanglement')==[] then
       experiment.entanglement=%F; //standard is no entanglement
   end;
   
   TJ=[1 1 1];  //standard value
   
   if grep(txt,'use_T_ts')==[] then
       experiment.use_T_ts=%f   //if it is non-existing set it to %f.
   elseif experiment.use_T_ts then
       TJ=experiment.T_ts;
   end;
   
  [e_sig, er_sig, en_sig, enr_sig]=e_epsilon(experiment);    
          
   if ~isvector(experiment.atom) then
      // go the easy way! It's only one spin-system

      S=experiment.atom;
      if length(S.J) == 1 then
          J(1)=S.J;J(2)=S.J;J(3)=S.J;
      else
          J=S.J;
      end

      matrxx=E'*Sx(S)*E;
      matrxy=E'*Sy(S)*E;
      matrxz=E'*Sz(S)*E;
      matrxu=S.U*E'*S1(S)*E;
      
      
      Smatrxx=J(1)*matrxx;
      Smatrxy=J(2)*matrxy;
      Smatrxz=J(3)*matrxz;
      Smatrxu=zeros(Smatrxx);

  else
    
      nr=experiment.position;
      nrj=experiment.jposition;
      natoms=size(experiment.atom);
      matrxx=sparse(1);
      for i=1:(nr-1),
          matrxx=matrxx.*.sparse(S1(experiment.atom(i)));
      end
      
      matrxy=matrxx;
      matrxz=matrxx;
      matrxu=matrxx;
      
      matrxx=matrxx.*.sparse(Sx(experiment.atom(nr)));
      matrxy=matrxy.*.sparse(Sy(experiment.atom(nr)));
      matrxz=matrxz.*.sparse(Sz(experiment.atom(nr)));
      matrxu=experiment.atom(nr).U*matrxu.*.sparse(S1(experiment.atom(nr)));
      
      for i=nr+1:natoms(2),
          tempm=sparse(S1(experiment.atom(i)));
          matrxx=matrxx.*.tempm;
          matrxy=matrxy.*.tempm;
          matrxz=matrxz.*.tempm;
          matrxu=matrxu.*.tempm;
      end
      
      matrxx=E'*matrxx*E;
      matrxy=E'*matrxy*E;
      matrxz=E'*matrxz*E;
      matrxu=E'*matrxu*E;
      
      if nrj==nr & ~experiment.entanglement then    //no entanglement
          
          J=experiment.atom(nrj).J;
          if length(J) == 1 then
              JT=J;
              J(1)=JT;J(2)=JT;J(3)=JT;
          end

          Smatrxx=J(1)*matrxx;
          Smatrxy=J(2)*matrxy;
          Smatrxz=J(3)*matrxz;
          Smatrxu=spzeros(Smatrxx);
          
      elseif ~experiment.entanglement then          //no entanglement
          Smatrxx=sparse(1);                        //and nrj<>nr
          
          for i=1:(nrj-1),
              Smatrxx=Smatrxx.*.sparse(S1(experiment.atom(i)));
          end
          
          Smatrxy=Smatrxx;
          Smatrxz=Smatrxx;
          
          Smatrxx=Smatrxx.*.sparse(Sx(experiment.atom(nrj)));
          Smatrxy=Smatrxy.*.sparse(Sy(experiment.atom(nrj)));
          Smatrxz=Smatrxz.*.sparse(Sz(experiment.atom(nrj)));
          for i=nrj+1:natoms(2),
              tempm=sparse(S1(experiment.atom(i)))
              Smatrxx=Smatrxx.*.tempm;
              Smatrxy=Smatrxy.*.tempm;
              Smatrxz=Smatrxz.*.tempm;
          end
          J=experiment.atom(nrj).J;
          if length(J) == 1 then
              JT=J;
              J(1)=JT;J(2)=JT;J(3)=JT;
          end

          Smatrxx=J(1)*E'*Smatrxx*E;
          Smatrxy=J(2)*E'*Smatrxy*E;
          Smatrxz=J(3)*E'*Smatrxz*E;
          Smatrxu=spzeros(Smatrxx);
          
      else      //entanglement
          Smatrxx=spzeros(matrxx);
          Smatrxy=spzeros(matrxx);
          Smatrxz=spzeros(matrxx);
          
          for j=1:natoms(2),
              Stmatrxx=sparse(1);                                
          
              for i=1:(j-1),
                  Stmatrxx=Stmatrxx.*.sparse(S1(experiment.atom(i)));
              end
              
              Stmatrxy=Stmatrxx;
              Stmatrxz=Stmatrxx;
              
              phi=0.0;//1.23;
              phase=exp(%i*phi*(j-nr))
          
              Stmatrxx=phase*Stmatrxx.*.sparse(Sx(experiment.atom(j)));
              Stmatrxy=phase*Stmatrxy.*.sparse(Sy(experiment.atom(j)));
              Stmatrxz=phase*Stmatrxz.*.sparse(Sz(experiment.atom(j)));
              for i=j+1:natoms(2),
                tempm=sparse(S1(experiment.atom(i)))
                Stmatrxx=Stmatrxx.*.tempm;
                Stmatrxy=Stmatrxy.*.tempm;
                Stmatrxz=Stmatrxz.*.tempm;
              end
              J=experiment.atom(j).J;
              if length(J) == 1 then
                JT=J;
                J(1)=JT;J(2)=JT;J(3)=JT;
              end

              Smatrxx=Smatrxx+J(1)*E'*Stmatrxx*E;
              Smatrxy=Smatrxy+J(2)*E'*Stmatrxy*E;
              Smatrxz=Smatrxz+J(3)*E'*Stmatrxz*E;
          end
          Smatrxu=spzeros(Smatrxx);
      end         
    
  end
    
    mone=ones(1,a(2));     

    matrxx=TJ(1)*matrxx;    //new in Version 1.6 we scale the tip sample scattering
    matrxy=TJ(2)*matrxy;
    matrxz=TJ(3)*matrxz;
    matrxu=-sign(TJ(1))*sign(TJ(2))*sign(TJ(3))*matrxu; ////new in Version 1.9.3 
           
    so1=list(matrxx,matrxy,matrxz,matrxu);
    so2=list(Smatrxx,Smatrxy,Smatrxz,Smatrxu);
    so3=so1;
    e=0;er=0;en=0;enr=0;
    for j=1:4,
        for k=1:4,
            for l=1:4,
                  if e_sig(j,k,l)<>0 then
                     e=e-e_sig(j,k,l)..
                     *(mone'.*.so3(l)(in,:)).*(so2(k)).'.*(mone.*.so1(j)(:,in));
                  end
                  if er_sig(j,k,l)<>0 then
                     er=er-er_sig(j,k,l)..
                     *(mone'.*.so3(l)(in,:)).*(so1(k)).'.*(mone.*.so2(j)(:,in));
                  end
                  if en_sig(j,k,l)<>0 then
                     en=en-en_sig(j,k,l)..
                     *(mone'.*.so1(l)(in,:)).*(so2(k)).'.*(mone.*.so3(j)(:,in));    
                  end
                  if enr_sig(j,k,l)<>0 then
                     enr=enr-enr_sig(j,k,l)..
                     *(mone'.*.so1(l)(in,:)).*(so3(k)).'.*(mone.*.so2(j)(:,in));
                  end
            end
        end
   end
   
   //e=clean(e);
   //er=clean(er)*0;
   //en=clean(en);
   //enr=clean(enr)*0;
   
endfunction

////////////////////////////////////////////////////////////////////////////////
// Bloch-Redfield functions
////////////////////////////////////////////////////////////////////////////////



function [out]=Redfield_coeff2(experiment)

    // This function calculates the Redfield coefficients due to the interaction 
    // of the spin system with the substrate.
    //
    // The calcuation follows mostly the path described in:
    //
    // Atom-Photon Interactions
    // Chapter IV
    // by Claude Cohen-Tannoudji, Jacques Dupont-Roc, and Gilbert Grynberg
    //
    // experiment has to contain:
    // 'experiment.ptip':        spin polarization of the tip
    // 'experiment.psample':     spin polarization of the sample
    // 'experiment.Eigenvec':    matrix of the eigenvectors of the spin ensemble
    // 'experiment.Eigenval':    matrix of the eigenvalues of the spin ensemble
    // 'experiment.T':           the temperature (in meV)
    // 'experiment.RedF_secular' either "yes" or 1 for a secular approach with 
    //                           with window 'experiment.RedF_secular_window'
    //                           or 'sinc' as an implementation of Fermi's golden rule
    // 'experiment.RedF_secular_window'
    // 'experiment.entanglement' should be set to %T to account for all scattering
    //                           processes in multispin systems
    //                        
    // 'a,b' are the coefficient of the density matrix $\sigma_{ab}$
    // 'out' is the matrix $R_{abcd}$ for the master equation:
    // $\frac{d}{dt}\sigma_{ab}(t)=\sum_{c,d}R_{abcd}\sigma_{ab}(t)$
    // 
    // note that the density matrix has to be given in the eigenstate basis of 
    // Hamiltonian !
    //
    // We do the calculation following IV.A, and IV.B, 
    // in particular we calculate equation IV.B.43
    // note that the terms with $\delta_{bd}$ and $\delta_{ac}$ are calculated
    // as "o1" and "o2". 
    // The remaining two terms 
    // $A_{ac}A_{db}[g(\tau)e^{i\omega_{ca}\tau}+g(-\tau)e^{i\omega_{bd}\tau}]$
    // are approximated as:
    // $M_{ac}M_{db}\times [f_{box}(\omega_{ca})+f_{box}(\omega_{bd})]$
    // with
    // $M_{ab}$ as the rate elements evaluated 2x2 times independetly.
    // $f_{box}(E)$ as the Fermi-Dirac Box function for states with an energy difference E
    // 
    
    if experiment.RedF_secular=="yes" then
        secular=%T;
        goldenR=%F;
        lt=%eps*10;
    elseif experiment.RedF_secular=="window" | experiment.RedF_secular==1 then
        secular=%T;
        goldenR=%F;
        lt=experiment.RedF_secular_window;
    elseif experiment.RedF_secular=="sinc" then
        secular=%F;
        goldenR=%T;
        lt=experiment.RedF_secular_window;
    else
        secular=%F;
        goldenR=%F;
    end 

    
    ve=experiment.Eigenval;
    len=length(ve);

    mone=ones(ve);mzero=0*mone;
    mmzero=zeros(len,len);
    out=zeros(len,len,len,len); 
    
     //calculate delta_E between the states
    ediff=mone.*.ve'-mone'.*.ve;    

    // calculate the electron probabilities in 2nd order
    proba=fbox(ediff./experiment.T)*experiment.T;
    
    tempexp=experiment;
    tempexp.ptip=tempexp.psample;
    atom_nr=length(experiment.atom);
    o=out;
    
    if experiment.entanglement then
        atm_cntr=[1:atom_nr];
    else
        atm_cntr=experiment.position;
    end
    
    for i=atm_cntr,     //sum over all sites
       tempexp.position=i;
       [M1,M2,M3,M4]=Rate2nd(tempexp);        
       
        for a=1:len,
            for b=1:len,
            
                secmatrx=ones(len,len);
    
                if secular==%T then
                    n=find(abs(ediff)>abs(ediff(a,b))+lt|abs(ediff)<abs(ediff(a,b))-lt);
                    secmatrx(n)=0;
                end
                
                if goldenR==%T then
                    ediff2=(abs(ediff)-abs(ediff(a,b)))/lt*%pi;
                    secmatrx=(sin(ediff2)./(ediff2)).^2;
                    secmatrx(isnan(secmatrx))=1;
                end
            
                o1=mzero;o2=mzero;
                for n=1:len,
                    o1=o1+..
                    (M1(n,a)*M1(n,:)'..
                    +M2(n,a)*M2(n,:)'..
                    +M3(n,a)*M3(n,:)'..
                    +M4(n,a)*M4(n,:)').*proba(n,:)';
                    o2=o2+..
                    (M1(n,b)*M1(n,:)'..
                    +M2(n,b)*M2(n,:)'..
                    +M3(n,b)*M3(n,:)'..
                    +M4(n,b)*M4(n,:)').*proba(n,:)';
                end 
                o11=mmzero;o11(:,b)=o1;
                o22=mmzero;o22(a,:)=o2';     
                //disp(string(a)+" "+string(b));
                o(:,:,a,b)=..
                ((M1(:,a)*M1(:,b)'..
                +M2(:,a)*M2(:,b)'..
                +M3(:,a)*M3(:,b)'..
                +M4(:,a)*M4(:,b)')..
                .*(proba(:,a)*mone'+mone*proba(:,b)')..
                -o11-o22).*secmatrx; 
            
            end
        end
        out=out+o;
    end
    
endfunction

function [out_ts,out_st]=Redfield_coeff_tunneling(experiment)

    // This function calculates the Redfield coefficients due to the interaction 
    // of the tunneling electrons with the spin system.
    //
    // The calculation follows mostly the path described in:
    //
    // Atom-Photon Interactions
    // Chapter IV
    // by Claude Cohen-Tannoudji, Jacques Dupont-Roc, and Gilbert Grynberg
    //
    // experiment has to contain:
    // 'experiment.ptip':     spin polarization of the tip
    // 'experiment.psample':  spin polarization of the sample
    // 'experiment.Eigenvec': matrix of the eigenvectors of the spin ensemble
    // 'experiment.T':        the temperature (in meV)
    // 'experiment.RedF_secular' either "yes" or 1 for a secular approach with 
    //                           with window 'experiment.RedF_secular_window'
    //                           or 'sinc' as an implementation of Fermi's golden rule
    // 'experiment.RedF_secular_window'
    // 'experiment.entanglement' should be set to %T to account for all scattering
    //                           processes in multispin systems
    //                        
    // 'a,b' are the coefficient of the density matrix $\sigma_{ab}$
    // 'out' is the matrix $R_{abcd}$ for the master equation:
    // $\frac{d}{dt}\sigma_{ab}(t)=\sum_{c,d}R_{abcd}\sigma_{ab}(t)$
    // 
    // note that the density matrix has to be given in the eigenstate basis of 
    // Hamiltonian !
    //
    // We do the calculation following IV.A, and IV.B, 
    // in particular we calculate equation IV.B.43
    // note that the terms with $\delta_{bd}$ and $\delta_{ac}$ are calculated
    // as "out1" and "out2". 
    // The remaining two terms 
    // $A_{ac}A_{db}[g(\tau)e^{i\omega_{ca}\tau}+g(-\tau)e^{i\omega_{bd}\tau}]$
    // are approximated as:
    // $M_{ac}M_{db}\times [f_{box}(\omega_{ca})+f_{box}(\omega_{bd})]$
    // with
    // $M_{ab}$ as the rate elements evaluated 2x2 times independetly.
    // $f_{box}(E)$ as the Fermi-Dirac Box function for states with an energy difference E
    // 
    
    if experiment.RedF_secular=="yes" then
        secular=%T;
        goldenR=%F;
        lt=%eps*10;
    elseif experiment.RedF_secular=="window" | experiment.RedF_secular==1 then
        secular=%T;
        goldenR=%F;
        lt=experiment.RedF_secular_window;
    elseif experiment.RedF_secular=="sinc" then
        secular=%F;
        goldenR=%T;
        lt=experiment.RedF_secular_window;
    else
        secular=%F;
        goldenR=%F;
    end 
    
    ve=experiment.Eigenval;
    len=length(ve);

    mone=ones(ve);mzero=0*mone;
    mmzero=zeros(len,len);
    out=zeros(len,len,len,len); 
    
     //calculate delta_E between the states
    ediff=mone.*.ve'-mone'.*.ve;    

    // calculate the electron probabilities in 2nd order
    U=experiment.xrange(1);
    
    proba_ts=fbox((ediff+U)./experiment.T)*experiment.T;
    proba_st=fbox((ediff-U)./experiment.T)*experiment.T;
    
    //proba=proba_ts+proba_st; this is only correct if no spin polarization! 
    
    tempexp=experiment;
    if experiment.use_T_ts then
        tempexp.atom(tempexp.position).J=experiment.T_ts;
    else
        tempexp.atom(tempexp.position).J=-sqrt(experiment.A);
        //we assume AFM coupling! Therefore the minus!
    end
    
    
    
    [M1TS,M2TS,M3TS,M4TS]=Rate2nd(tempexp);
    
    //flip tip and sample
    ptip=tempexp.ptip;
    tempexp.ptip=tempexp.psample;
    tempexp.psample=ptip;
    
    [M1ST,M2ST,M3ST,M4ST]=Rate2nd(tempexp);
    
   
    for a=1:len,
        for b=1:len,
        
            secmatrx=ones(len,len);

            if secular==%T then
                n=find(abs(ediff)>abs(ediff(a,b))+lt|abs(ediff)<abs(ediff(a,b))-lt);
                secmatrx(n)=0;
            end
            
            if goldenR==%T then
                ediff2=(abs(ediff)-abs(ediff(a,b)))/lt*%pi;
                secmatrx=(sin(ediff2)./(ediff2)).^2;
                secmatrx(isnan(secmatrx))=1;
            end
        
            o1=mzero;o2=mzero;o3=mzero;o4=mzero;
            for n=1:len,
                o1=o1+(M1TS(n,a)*M1TS(n,:)'+..
                       M2TS(n,a)*M2TS(n,:)'+..
                       M3TS(n,a)*M3TS(n,:)'+..
                       M4TS(n,a)*M4TS(n,:)').*proba_ts(n,:)';
                o2=o2+(M1TS(n,b)*M1TS(n,:)'+..
                       M2TS(n,b)*M2TS(n,:)'+..
                       M3TS(n,b)*M3TS(n,:)'+..
                       M4TS(n,b)*M4TS(n,:)').*proba_ts(n,:)';
                       
                o3=o3+(M1ST(n,a)*M1ST(n,:)'+..
                       M2ST(n,a)*M2ST(n,:)'+..
                       M3ST(n,a)*M3ST(n,:)'+..
                       M4ST(n,a)*M4ST(n,:)').*proba_st(n,:)';
                o4=o4+(M1ST(n,b)*M1ST(n,:)'+..
                       M2ST(n,b)*M2ST(n,:)'+..
                       M3ST(n,b)*M3ST(n,:)'+..
                       M4ST(n,b)*M4ST(n,:)').*proba_st(n,:)';

            end 
            o11=mmzero;o11(:,b)=o1;
            o22=mmzero;o22(a,:)=o2';
            o33=mmzero;o33(:,b)=o3;
            o44=mmzero;o44(a,:)=o4';
            
                 
            //disp(string(a)+" "+string(b));
            out_ts(:,:,a,b)=..
            ((M1TS(:,a)*M1TS(:,b)'+M2TS(:,a)*M2TS(:,b)'+M3TS(:,a)*M3TS(:,b)'+M4TS(:,a)*M4TS(:,b)')..
            .*(proba_ts(:,a)*mone'+mone*proba_ts(:,b)')-o11-o22)..
            .*secmatrx; 
            out_st(:,:,a,b)=..
            ((M1ST(:,a)*M1ST(:,b)'+M2ST(:,a)*M2ST(:,b)'+M3ST(:,a)*M3ST(:,b)'+M4ST(:,a)*M4ST(:,b)')..
            .*(proba_st(:,a)*mone'+mone*proba_st(:,b)')-o33-o44)..
            .*secmatrx; 
        
        end
    end

    
endfunction

function [out]=Redfield_eshift(experiment)

    tempexp=experiment;
    tempexp.ptip=tempexp.psample;
    
    ve=experiment.Eigenval;
    ve=ve-min(real(ve));
    atom_nr=size(experiment.atom,2);
    
    if experiment.entanglement then
        atm_cntr=[1:atom_nr]
    else
        atm_cntr=experiment.position
    end
    
    M=zeros(ve.*.ve');
    for i=atm_cntr,     //sum over all sites
        tempexp.position=i;
        M=M+Rate2nd2(tempexp);
    end
   
    xrange=max([2*ve+20*experiment.T;20*experiment.T]);
    xpoints=floor(max([xrange/experiment.T;1000]));
    x=linspace(-xrange,xrange,xpoints); //integration grid 
    atom=experiment.atom(experiment.position);

     //calculate delta_E between the states
    ediff=ones(1:size(ve,1))'.*.ve'-ones(1:size(ve,1)).*.ve;
    mzero=zeros(ve);
    proba=[];
    
    for i=1:size(ediff,1),
        for j=1:size(ediff,2),
            proba(i,j)=sum(fstep_simple((x-ediff(i,j))/experiment.T)...
            .*(1-fstep_simple(x/experiment.T)).*ln_t2(x,atom.w,experiment.T,experiment.lt)');
            //proba(i,j)=sum(fstep_simple((x-ediff(i,j))/experiment.T).*(1-fstep_simple(x/experiment.T)));
        end
    end
    proba=real(proba*2*xrange/(xpoints-1)); //scale
    // calculate the scattering probabilities in 2nd order
    //proba=fbox(ediff./experiment.T)*experiment.T;
    out=mzero';
    for n=1:length(ve),
        out=out+M(n,:).*proba(n,:);
    end
    out=out*(atom.J^2);
endfunction


function [out,I,occ,ltime,trans_matrix,rnk]=Redfield_dynamic2(experiment,dm,t)

    global experiment;
    ve=experiment.Eigenval;
    m=experiment.Eigenvec;
    len=(size(dm,1));

    //calculate delta_E between the states and with this the Bohr frequencies
    // 1meV = 0.24179905THz ->
    ediff=-%i*(ones(1:size(ve,1))'.*.ve'-ones(1:size(ve,1)).*.ve)*0.24179905*2*%pi;
    
    txt=getfield(1,experiment);
    if grep(txt,'RedF_coeff')==[] then
        experiment.RedF_coeff=[];
    end

    //RedF_coeff has to be calculated only once per spectrum
    if experiment.RedF_coeff==[] then   
       experiment.RedF_coeff=Redfield_coeff2(experiment);
    end
    [tun_ts,tun_st]=Redfield_coeff_tunneling(experiment);
    RF=experiment.RedF_coeff+tun_ts+tun_st;
    

     //reduce RF to the rate matrix of the diagnal elemets of the dm
    for i=1:len,
        rate2(:,i)=diag(RF(:,:,i,i));
    end
    //the inverse of the lifetime is:
    ltime=sum(rate2-rate2.*eye(rate2),1)';
    
    
    //rescale RF for having t in [ps]:
    //energy is in units of [meV]
    //the conductance in units of G_0=77.48 uS
    //the charge of an electron is e_0=1.6e-19 C
    //the scaling factor is:
    //(meV*G_0)/e_0=0.48359788 THz
    RF=0.48359788*matrix(RF,length(ve)^2,length(ve)^2);
    sz=size(dm);
    dm=matrix(dm,length(dm),1); //make the dm vector
    ediff=matrix(ediff,length(ediff),1); //make the ediff vector
    ediff=diag(ediff);                  //and diagonal
    
    J2_ts=experiment.A;     //experiment.A is J^2 of the tip-sample junction
  
    //calculate delta_E between the different states
    ediff2=ones(1:size(ve,1))'.*.ve'-ones(1:size(ve,1)).*.ve;
  
    rate_ts=Rate2nd2(experiment);
    rate_st=rate_ts';
    U=experiment.xrange;

    rate2_ts=J2_ts*fbox((ediff2+U)/experiment.T)*experiment.T.*rate_ts;
    rate2_st=J2_ts*fbox((ediff2-U)/experiment.T)*experiment.T.*rate_st;
    
    //trans_matrix is now a vector of the exponent for the density matrix. 
    trans_matrix=RF+ediff;
    
       
    for i=1:length(t),
        disp("round :"+string(i));
        to=expm((trans_matrix)*t(i));
        
        o=to*dm;    
        o=matrix(o,sz);
        out(:,:,i)=o-%i*diag(imag(diag(o))); //remove imags from the matrix diagonal
        stst=real(diag(o));
        occ(i,:)=stst';
        rnk(i)=or(stst<%eps*1000)*1+1;
        I(i)=real(sum(rate2_ts*stst)-sum(rate2_st*stst));
    end   
endfunction

function out=Redfield_ss(experiment)
    ve=experiment.Eigenval;
    ve=ve-min(real(ve));
    
    RF=zeros(length(ve),length(ve),length(ve),length(ve));
    for i=1:length(ve),
        for j=1:length(ve),
            RF(:,:,i,j)=Redfield_coeff(experiment,i,j);
        end
    end
    RF2=matrix(RF,length(ve)^2,length(ve)^2);
    RF3=RF2-eye(RF2).*RF2-diag(sum(RF2-eye(RF2).*RF2,1));
    
    out=kernel(RF2);
    
    out=matrix(out(:,1),length(ve),length(ve));
    out=out/trace(out);
    
endfunction
