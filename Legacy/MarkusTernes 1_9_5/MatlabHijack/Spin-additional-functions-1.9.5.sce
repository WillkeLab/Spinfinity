////////////////////////////////////////////////////////////////////////////////
//
//    Spin-additional-functions-1.9.5.sce
//
//    Stuttgart, Aachen, Juelich Febrary 2021
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

funcprot(0);
ieee(2)

////////////////////////////////////////////////////////////////////////////////
//some usefull functions
////////////////////////////////////////////////////////////////////////////////

//Appelbaum's functions 

function[e]=AvrgM(S,delta,T)
    // This is the Brillouin function 
    // for example, for S=1/2 it is the same as:
    // +S*Occupation(d,1)+(-S)*Occupation(d,2) with d=[-delta 0; 0 delta]
    e=(2*S.S+1)/(2*S.S)*coth((2*S.S+1)/(2*S.S)*delta/(2*T))..
    -1/(2*S.S)*coth(1/(2*S.S)*delta/(2*T));
endfunction

////////////////////////////////////////////////////////////////////////////////
//some special analytical functions
////////////////////////////////////////////////////////////////////////////////

function[e]=ln_t2(x,e0,T,lt)
    // 'e' is the temperature folded log function
    //
    //   +e0/
    //     [
    //   e=I  f(e',T)/(e-e'+i lt) de' * f'(e,T)
    //     ]
    // -e0/
    //
    // with f(e,T) as the Fermi-Dirac distribution and * as the convolution
    //
    // 'x' is a vector of tunneling voltages
    // 'e0' is the half-bandwidth of the substrate electrons
    // 'T' is the temperature 
    // 'lt' is the lifetime broadening
       
    xspan=x(1)-x($);
    i=2*length(x); // i has now 2 times the size of x
    y=linspace(xspan,-xspan,i+1); //we need the expanded range of y for the correct convolution.
    g=log((e0+abs(y))./(y+lt*%i))+%pi*%i/2; //take the logarithm of this larger range
    z=x/T; //normalize the energy (x) with the temperature (T) for the Fermi-Dirac distribution
    k=find(abs(z)<1e-3); //make a list k with values close to 0 which will produce wrong results 
    ez=exp(z);   
    f=-((ez+ez.*(z-1)).*(ez-1).^-2-2*ez.*(ez.*(z-1)+1).*(ez-1).^-3);
    f(k)=1/6;   // the values of the list (k) are at the maximum of 1/6
    f(isnan(f))=0;  //out-of-range values are set to zero
    e=2*conv2(f(:),g(:),"same")./(T*i).*abs(xspan); //perform the convolution
endfunction

function[e]=ln_t2int(x,e0,T,lt)
    // 'e' is the integrated temperature folded log function
    //
    //     x/
    //     [
    //   e=I  ln_t2(x')*dx'
    //     ]
    //   0/
    //
    // 'x' is a vector of tunneling voltages
    // 'e0' is the half-bandwidth of the substrate electrons
    // 'T' is the temperature 
    // 'lt' is the lifetime broadening
    
    xspan=x(1)-x($);
    i=2*length(x); // i has now 2 times the size of x
    y=linspace(+xspan,-xspan,i+1);//we need the expanded range of y for the correct convolution
    
    g=(y).*log((e0+abs(y))./(y+lt*%i))...
    +sign(y).*((e0*log(e0+abs(y)))-e0*log(e0)...
    +lt*%i*log(-lt*%i+abs(y))-lt*%i*log(-lt*%i));//+%pi*%i/2;
    z=x/T; //normalize the energy (x) with the temperature (T) for the Fermi-Dirac distribution
    k=find(abs(z)<1e-3); //make a list k with values close to 0 which will produce wrong results 
    ez=exp(z);
    f=-((ez+ez.*(z-1)).*(ez-1).^-2-2*ez.*(ez.*(z-1)+1).*(ez-1).^-3);
    f(k)=1/6;  // the values of the list (k) are at the maximum of 1/6
    f(isnan(f))=0; //out-of-range values are set to zero
    e=2*conv(f,g,"same")./(T*i).*abs(xspan); //perform the convolution
    // e is now the integral of ln_t2
    
endfunction

function[e]=ln_t2int_2(x,e0,T,lt)
    // 'e' is the integrated temperature folded log function
    //
    //     x/
    //     [
    //   e=I  ln_t2(x')*dx'
    //     ]
    //   0/
    //
    // 'x' is a vector of tunneling voltages
    // 'e0' is the half-bandwidth of the substrate electrons
    // 'T' is the temperature 
    // 'lt' is the lifetime broadening
    
    
    g=(x).*log((e0+abs(x))./(x+lt*%i))...
    +sign(x).*((e0*log(e0+abs(x)))-e0*log(e0)...
    +lt*%i*log(-lt*%i+abs(x))-lt*%i*log(-lt*%i));//+%pi*%i/2;
    z=x/T; //normalize the energy (x) with the temperature (T) for the Fermi-Dirac distribution
    f=abs(fstep(z)-0.5);
    e=g.*f;
    // e is now the integral of ln_t2
    
endfunction


function [e,en]=int_ln_step(x,e0,T,lt,stpos)
    //
    // vd:   peak function
    // v:  integrated peak function
    // u:   step function
    // ud:  differentiated step function
    //
    // int (u*v')=u*v-int(u'*v)
    //
    // this function is NOT working yet...
    // sorry dude... ...math is too hard for me today...
    // if someone knows the result of the 2nd part on the right side...
    // please let me know...
    
    vd=ln_t2(x,e0,T,lt); //get vd
    v=ln_t2int(x,e0,T,lt); //get v
    u=fstep((x-stpos)/T); //get u
    
    z=(x-stpos)/T;
    k=find(abs(z)<1e-3); //values arround zero produce wrong results 
    ez=exp(z);    //k=find(ez==1); ez(k)=1+10^-10;z(k)=10^-10;
    ud=-((ez+ez.*(z-1)).*(ez-1).^-2-2*ez.*(ez.*(z-1)+1).*(ez-1).^-3);
    ud(k)=1/6;
    ud(isnan(ud))=0;
    
    
    e=1*v.*u;
    en=v.*ud;   //how do we smartly integrate this part??
    
endfunction

function[e]=folding_sinc2(x,fx,T)
    // 'e' is the temperature broaded data of fc
    //
    //   +inf/
    //      [
    // e(x)=I fx(x+t') * sinc^2(t'*T)
    //      ]
    // -inf/
    //
    // with f(e,T) as the Fermi-Dirac distribution and * as the convolution
    //
    // 'x' is a vector of tunneling voltages
    // 'fx' is the external data which will be broadened
    // 'T' is the temperature 
       
    xspan=x(1)-x($);
    i=2*length(x); // i has now 2 times the size of x
    y=linspace(xspan,-xspan,i+1); //we need the expanded range of y for the correct convolution.
    g=(sin(y*T).^2)./((y*T).^2);
    g(isnan(g))=1;
    g=g./sum(g);
    e=conv2(g(:),fx(:),"same"); //perform the convolution
endfunction

function [e]=Occupation(ve, T)
  // calculate occupation using the Boltzmann distribution
  // ve are the eigenvalues (in meV) 
  // T is the effective temperature (in meV)
  ve=real(ve);
  ve=ve-min(ve);
  if T<>0 then //check that T<>0
      e=exp(-ve/T);
      e=e/sum(e); 
  else         //if T=0 then distribute equaly over the energetically lowest states
      bb=size(ve); 
      e(1:bb(1))=0; 
      k=min(ve);
      e(find(ve==k))=1;
      e=e./sum(e)
  end
endfunction

function [y]=fstep_simple(x)
   // Fermi-Dirac step function
   y=(exp(x)+1).^-1;
endfunction

function [y]=fstep(x)
    // step function; doubly temperature broadened
    x=clean(x,1e-5); // find all x(i)~0
    z=exp(x);
    y=(1+(x-1).*z)./((z-1).^2);
    y(x==0)=0.5;
    y(isnan(y))=0; //remove NaN values because of x too large and set them to 0
endfunction

function [y]=fbox(x)
   // Fermi-Dirac box function
   //
   //   +inf/
   //      [
   //    y=I  f(e')*[1-f(e'+x)] de'
   //      ]
   // -inf/
   //
   // with f(e,T) as the Fermi-Dirac distribution 
   //
   x=clean(x,1e-8); //set values <+-1e-8 to zero
   y=x./(1-exp(-x));
   y(isnan(y))=1;
endfunction

function [e]=State(t,n)
    // Gives a list of the mz value of the ensemble t of spins 
    // at the eigenstate number n.
    // State([Fe,Co,...],n)
    
    b=size(t,2);     // determine the number of coupled spins
    e=zeros(length(n),b);
    for i=1:b,
        m=1;
        for j=1:(i-1), 
            m=m.*.sparse(S1(t(j))); 
        end;
        m=m.*.sparse(Sz(t(i)));
        for j=(i+1):b, 
            m=m.*.sparse(S1(t(j))); 
        end;
        m=full(diag(m));
        for k=1:length(n),
            e(k,i)=m(n(k));
        end
    end;
endfunction

function [e]=Dde(S)
    //calculate D and E from Dxx,Dyy,Dzz
    S.D=S.Dzz-0.5*(S.Dxx+S.Dyy);
    S.E=0.5*(S.Dxx-S.Dyy);
    e=S;
endfunction

function [e]=Dxyz(S)
    //calculate Dxx,Dyy,Dzz from D and E
    S.Dzz=S.D;
    S.Dxx=S.Dzz-S.D+S.E;
    S.Dyy=S.Dzz-S.D-S.E;
    e=S;
endfunction

function [e]=Stev2DE(S)
    //calculate D and E from the Stevensoperator
     Stev=S.Stevens;
     [a,b]=size(Stev);
     if a>0 & b>0 then 
         S.D=3*Stev(1,1);
     end
     if a>0 & b>2 then
         S.E=Stev(1,3);
     end
     e=S;
endfunction

function [e]=DE2Stev(S)
    //calculate Stevensoperator from D and E 
     S.Stevens(1,1)=S.D/3;
     S.Stevens(1,3)=S.E;
     e=[S];
endfunction

function [err]=savedat(name,x,y,text)
   //save the data (x,y)
   rhs=argn(2);
   if (rhs>4) then
     error(58); //incorrect number of arguments
     abort;
   end
  
   if (rhs==3) then //it's fine if there's no text provided
     text=[]; 
   end
  
   if type(name)<>10 then 
       error(55,1); 
       abort;
   end
   [xx,xy]=size(x);
   [yx,yy]=size(y);
 
   if (xx~=1) & (xy~=1) then 
       error(56,2);
       abort; //x is not vector
   end;
   if  xx==1 then 
       x=x.'; 
       y=y.'; 
       end;

   fo=mopen(name,'w');

   mfprintf(fo,'# Data calculated with a routine written by\n#\n');
   mfprintf(fo,'# Markus Ternes (markus.ternes@gmail.com)\n');
   mfprintf(fo,'#\n');
   mfprintf(fo,'# The calculation was saved on:'+date()+'\n');
   time=getdate();
   mfprintf(fo,'# at:%i:%i:%i\n',time(7),time(8),time(9));
   mfprintf(fo,'#\n');
  
   if text<>[] then
      for i=1:size(text,1),
        mputstr('#'+string(text(i)),fo);
        mfprintf(fo,'\n');  
      end
   end

   for n=1:size(y,2),
     mfprintf(fo,'# set number %f\n', n);
     mfprintf(fo,'#\n');

     for i=1:size(y,1),
        mfprintf(fo,'%e\t%e\n',x(i),y(i,n));
     end;
     mfprintf(fo,'\n');
   end,
   err=mclose(fo);
endfunction

function [c]=crossp(a,b)
    // calculates the crossproduct c=a x b;
    c(1)=a(2)*b(3)-a(3)*b(2);
    c(2)=a(3)*b(1)-a(1)*b(3);
    c(3)=a(1)*b(2)-a(2)*b(1);
endfunction

function [e]=vNentropy(rho)
    // calculates the von Neumann entropy of the density matrix rho 
    if clean(trace(rho^2)-trace(rho))==0 then
        e=0
    else
       a=spec(rho);
       b=log(a);
       b(isnan(b))=0;   //make 0*log(0) equal to 0
       b(isinf(b))=0;   //make 0*log(0) equal to 0
       e=-sum(b.*a);
       e=real(e);
    end
endfunction

function [e]=negativity(rho,t)
    // calculates the "negativity" of the spin ensemble t,
    // rho is the state vector or the density matrix.
    //
    // The negativity is a good measure for entanglement for S=1/2 and S=1
    // systems. 
    // It is calculated by summing the negative eigenvalues of the partially, 
    // with respect to the subsystem of the i-th spin, transposed density matrix
    //
    // see p ex.:
    // A. Peres, Phys. Rev. Lett. 77 (1996) 1413.
    // G. Vidal, R.F. Werner, Phys. Rev. A 65 (2002) 032314.
    // P. Horodecki, Phys. Lett. A 232 (1997) 333.
    //
    // output [e] is a vector of the same size as t.
    
    // check if rho is state-vector or density matrix
    if or(size(rho)==1) then
       rho=rho*rho';    // convert state-vector to density matrix
    end;
    
     b=size(t,2);       // determine the number of coupled spins
     for i=1:b,
         pt=ptranspose(rho,t,i);
         sp=spec(pt);          
         // We solve the full eigenvalue problem. This limits the size of rho.
         // better would be when we use:
         // sp=eigs(pt,speye(pt),50) 
         // however, what would be a good limit here?
         e(i)=sum(abs(real(sp))-real(sp))/2;
     end

endfunction


////////////////////////////////////////////////////////////////////////////////
// Spectrum calculation routines
////////////////////////////////////////////////////////////////////////////////

function [x,y]=spec2(experiment)
    // function to calculate the spectrum in 2nd order
    // experiment has to contain at least the following information:
    //
    // experiment.T         (temperature)
    // experiment.xrange    (xrange or list of x-values)
    // experiment.ptip      (tip spin-polarizaion)
    // experiment.psample   (sample Spin-polarization)
    // experiment.atom      (either the parameters of an atom or a list of atoms)
    // experiment.Eigenvec  (at least two eigenvectors of the Hamiltonian)
    // experiment.Eigenval  (at least two eigenvalues of the Hamiltonian)
    // experiment.position  (the position of the tip when we have a list of atoms)
    //
    // additionally, the following global variables have to be set:
    //
    // %savedata=%T or %F   (if the data should be saved in files directly)
    // %precission=0.001    (lower limit of occupation)
    // %plotspec=%T or %F   (if the data should be plotted directly)
    // 
    // 'x' is a list of the tunneling voltages
    // 'y' is a list of the dI/dV values 
    
    ve=real(experiment.Eigenval);

    ve=ve-min(ve);  // Energy = 0 is the ground state.
    T=abs(experiment.T);
    
    occ=Occupation(ve,T);
    
    txt=getfield(1,experiment);
    if grep(txt,'func_tip')==[] then
        experiment.func_tip=0;
    end
    
    if grep(txt,'spec_length')==[] then 
        spec_length=1000;
    else
        spec_length=experiment.spec_length;
    end
    
    if ~experiment.func_tip then
        rate=Rate2nd2(experiment);     //normal transtion rate calculation
    elseif experiment.func_tip==1 then
        //calculate the transiton rate matrix with a functionalized (dark) tip 
        rate=Rate2nd2_func_tip(experiment);
    else
        //calculate the transiton rate matrix with a functionalized (bright) tip 
        rate=Rate2nd2_func_tip_2(experiment);
    end
        
    if isvector(experiment.xrange) then x=experiment.xrange,
        else x=linspace(abs(experiment.xrange),-abs(experiment.xrange),spec_length)';
    end;
    
    y=0;
    a=size(experiment.Eigenvec);
   
    for i=1:a(2),           //loop over all initial staes i 
        if occ(i)>%precission then
            for j=1:a(2),   //loop over all final staes j 
                ep=fstep((ve(j)-ve(i)-x)/T); //stepfunction at positive bias
                en=fstep((ve(j)-ve(i)+x)/T); //stepfunction at positive bias
                ytemp=occ(i)*(rate(j,i)*ep+rate(i,j)*en);
                if %savedata then 
                    csvWrite([x real(ytemp)],..
                    "data"+string(i)+"-"+string(j)+"SF.dat"," ");
                end
                y=y+ytemp;
            end,
        end,
    end;
    
    if %plotspec then 
        gcf(); 
        plot(x,y), 
    end;
    
endfunction


function [x,y,yr]=spec3(experiment,in)
    // function to calculate the spectrum in 3rd order
    // experiment has to contain at least the following information:
    //
    // experiment.T         (temperature)
    // experiment.xrange    (xrange or list of x-values)
    // experiment.lt        (lifetime broadening)
    // experiment.ptip      (tip spin-polarizaion)
    // experiment.psample   (sample spin-polarization)
    // experiment.atom      (either the parameters of an atom or a list of atoms)
    // experiment.Eigenvec  (at least two eigenvectors of the Hamiltonian)
    // experiment.Eigenval  (at least two eigenvalues of the Hamiltonian)
    // experiment.position  (the position of the tip when we have a list of atoms)
    // experiment.jposition (the position where the sample-sample transition takes place)
    //                      (experiment.jposition will be set to experiment.position if not set)
    //
    // additionally, the following global variables have to be set:
    //
    // %savedata=%T or %F   (if the data should be saved in files directly)
    // %precission=0.001    (lower limit of evaluated matrix elements)
    // %plotspec=%T or %F   (if the data should be plotted directly)
    // 
    // 'in' is the initial state number of the calculation
    // 'x' is a list of the tunneling voltage
    // 'y' is a list of the dI/dV values for 'normal' tunneling
    // 'yr' is a list of the dI/dV values for 'time reversed' tunneling
    
    txt=getfield(1,experiment);
    if grep(txt,'jposition')==[] then 
        experiment.jposition=experiment.position;
    end;

    if grep(txt,'spec_length')==[] then 
        spec_length=1000;
    else
        spec_length=experiment.spec_length;
    end

    ve=experiment.Eigenval;

    ve=ve-min(real(ve));

    T=abs(experiment.T);
    lt=experiment.lt;
    w=experiment.atom(experiment.jposition).w;

    occ=Occupation(real(experiment.Eigenval),T);

    if ~experiment.func_tip then                        //normal tip
        [rate,rater,raten,ratenr]=M35(experiment,in);   //go standart way
        
    elseif experiment.func_tip==1 then                  //"dark" tip
        [rate,rater,raten,ratenr]=M35_func_tip(experiment,in); 

    else                                                //"bright" tip
        [rate,rater,raten,ratenr]=M35_func_tip_2(experiment,in);
    end

    if isvector(experiment.xrange) then x=experiment.xrange,
        else x=linspace(abs(experiment.xrange),-abs(experiment.xrange),spec_length)';
    end;
    
     if experiment.use_T_ts then 
        prec=%precission^2*sum(experiment.T_ts.^2)^2/9;   //rescale precission
    else
        prec=%precission^2;
    end
   
    y=zeros(x);yn=zeros(x);yr=zeros(x);ynr=zeros(x);
    a=size(experiment.Eigenvec);
    for mid=1:a(2),
       for fin=1:a(2),
            if abs(rate(mid,fin))+abs(raten(mid,fin))+...
                abs(rater(mid,fin))+abs(ratenr(mid,fin))>prec then
                tlog1=ln_t2(ve(mid)-ve(in)+x,w,T,lt);
                tlog2=ln_t2(ve(mid)-ve(in)-x,w,T,lt);
                
                ep=fstep((ve(fin)-ve(in)-x)/T); //stepfunction at positive bias
                en=fstep((ve(fin)-ve(in)+x)/T); //stepfunction at negative bias
                
                ytemp=-occ(in)*2*real(rate(mid,fin)*ep.*tlog2);
                ytempn=-occ(in)*2*real(raten(mid,fin)*en.*tlog1);

                //temporal reverse                
                ytempr=-occ(in)*2*real(rater(mid,fin)*ep.*tlog1);
                ytempnr=-occ(in)*2*real(ratenr(mid,fin)*en.*tlog2);
            
                if %savedata then 
                    csvWrite([x (ytemp+ytempn)],..
                    "data"+string(in)+"-"+string(mid)+"-"+string(fin)+".dat"," ");
                    csvWrite([x (ytempr+ytempnr)],..
                    "data"+string(in)+"-"+string(mid)+"-"+string(fin)+"r.dat"," ");
                end;

                y=y+ytemp+ytempn;
                yr=yr+ytempr+ytempnr;

            end;
        end;
    end;
    if %plotspec then 
        plot(x,[real(y),real(yr),real(y+yr)]), 
    end;
endfunction

function [out,outn]=spec3rate(experiment,varargin)
    // function to calculate the spectrum in 3rd order WITH RATE EQUATIONS
    // experiment has to contain at least the following informations:
    //
    // experiment.T         (temperature)
    // experiment.xrange    (xrange or list of x-values)
    // experiment.lt        (lifetime broadening)
    // experiment.ptip      (tip spin-polarizaion)
    // experiment.psample   (sample Spin-polarization)
    // experiment.atom      (either the parameters of an atom or a list of atoms)
    // experiment.Eigenvec  (at least two eigenvectors of the Hamiltonian)
    // experiment.Eigenval  (at least two eigenvalues of the Hamiltonian)
    // experiment.position  (the position of the tip when we have a list of atoms)
    // experiment.jposition (the position where the sample-sample transition takes place)
    //                      (experiment.jposition will be set to experiment.position if not set)
    //
    // additionally, the following global variables have to be set:
    //
    // %precission=0.001    (lower limit of evaluated matrix elements)
    // 
    // 'out' is a tensor with [a,a,spec_length] elements 
    // (a=number of eigenstates)
    // (spec_length as defined in experiment.spec_length or 1000)
    // and the following meaning:
    // current_ts=out[final-state,initial-state,voltage_Step]
    // current_st=outn[final-state,initial-state,voltage_Step]
    //
    // for example: 
    // out(2,1,100) describes the probability that an electron tunneling from
    // tip to sample exites, due to the applied voltage of V=x(100), the spin
    // system from its state 1 to the state 2 via 3rd order processes.
    // 
    
    info=%T;
    [lhs,rhs]=argn(0); 
    if rhs > 1 then
        info=varargin(1);
    end
    
    if ~exists('%stopcalc') then
        %stopcalc=%F;
    end
    
    ve=experiment.Eigenval;
    ve=ve-min(real(ve));        //set minimum Eigenstate energy to zero
    
    T=abs(experiment.T);
    lt=experiment.lt;
    w=experiment.atom(experiment.jposition).w;

    a=size(experiment.Eigenvec);
    
    if isvector(experiment.xrange) then 
        rnge=max(abs(experiment.xrange)); 
        if experiment.xrange(1)<0 then 
            rnge=-rnge;
        end
        spec_length=length(experiment.xrange);
        rnge=2*rnge/spec_length*(spec_length-0.5)
    else
        spec_length=experiment.spec_length;
        rnge=2*experiment.xrange/spec_length*(spec_length-0.5);
    end;

    x=linspace(rnge,-rnge,2*spec_length)';  //define symmetrically to zero the convolution range
    
    out=zeros(a(2),a(2),spec_length); //initilize output matrix
    outn=zeros(a(2),a(2),spec_length); //initilize output matrix
    
    yspan=(x(1)-x($))/2;ycenter=(x(1)+x($))/2;
    i=length(x); // i has now 2 times the size of x
    y=linspace(ycenter+yspan,ycenter-yspan,i);

    xkernel=1-fstep_simple(y/T);
  
    if info then
        winH=waitbar('Calculating Spectrum');  
    end
    
    for in=1:a(2),
       if experiment.func_tip == 0 then
          [rate,rater,raten,ratenr]=M35(experiment,in);
       elseif experiment.func_tip == 1 then           
          [rate,rater,raten,ratenr]=M35_func_tip(experiment,in);
        
          if experiment.entanglement then  //allow also scattering on the tip
             experiment.jposition=1;
             experiment.entanglement=%f;
             [brate,brater,braten,bratenr]=M35_func_tip(experiment,in);
          
             rate=rate+brate;
             rater=rater+brater;
             raten=raten+braten;
             ratenr=ratenr+bratenr;
             experiment.entanglement=%t;
          end
       else
          disp("Rate equations with spectroscopically active tip are not yet implemented!");
          abort;
       end;
             
       if ~%stopcalc && info then
           waitbar((in-1)/a(2),['Calculating Spectrum';..
           ' ';'Evaluate State '+string(in)+'('+string(a(2))+..
           ') with Spin-Spin Interaction on Atom '+string(experiment.jposition)],winH);
       elseif info then
           close(winH);
           return;
       end
       
        if experiment.use_T_ts then 
            prec=%precission*sum(experiment.T_ts.^2)^2/9;   //rescale precission
        else
            prec=%precission;
        end
    
       
       
       for fin=1:a(2),
           yp=zeros(x);ypr=zeros(x);
           yn=zeros(x);ynr=zeros(x);
           for mid=1:a(2),
              if abs(rate(mid,fin))+abs(raten(mid,fin))+...
                abs(rater(mid,fin))+abs(ratenr(mid,fin))>prec then
                
                
                tlog1=ln_t2(ve(mid)-ve(in)+x,w,T,lt);
                tlog2=ln_t2(ve(mid)-ve(in)-x,w,T,lt);
                ep=fstep_simple((ve(fin)-ve(in)-x)/T);  //stepfunction at pos bias
                en=fstep_simple((ve(fin)-ve(in)+x)/T);  //stepfunction at neg bias
                
                ytemp=2*real(rate(mid,fin)*ep.*tlog2);
                ytempn=2*real(raten(mid,fin)*en.*tlog1);
                
                //temporal reverse                
                ytempr=2*real(rater(mid,fin)*ep.*tlog1);
                ytempnr=2*real(ratenr(mid,fin)*en.*tlog2);
                
                yp=yp+ytemp;
                yn=yn+ytempn;
                ypr=ypr+ytempr;
                ynr=ynr+ytempnr;
                
              end;
           end; 
          
           z=conv2(yp(:)+ypr(:),xkernel(:),"same")/spec_length*(max(x));
           out(fin,in,:)=z(round(0.5*spec_length)+1:round(1.5*spec_length));
           zn=conv2(yn($:-1:1)+ynr($:-1:1),xkernel(:),"same")/spec_length*(max(x));
           outn(fin,in,:)=zn(round(1.5*spec_length):-1:round(0.5*spec_length)+1);
           
       end;
    end;

    if info then
        close(winH);
    end
    out=out;
    outn=outn;
endfunction

function [out]=spec3rate_ss(experiment)
    // function to calculate the sample-sample transission probabilities 
    // in 3rd order 
    // experiment has to contain at least the following informations:
    //
    // experiment.T         (temperature)
    // experiment.lt        (lifetime broadening)
    // experiment.ptip      (tip spin-polarizaion)
    // experiment.psample   (sample Spin-polarization)
    // experiment.atom      (either the parameters of an atom or a list of atoms)
    // experiment.Eigenvec  (at least two eigenvectors of the Hamiltonian)
    // experiment.Eigenval  (at least two eigenvalues of the Hamiltonian)
    // experiment.position  (the position of the tip when we have a list of atoms)
    // experiment.jposition (the position where the sample-sample transition takes place)
    //                      (experiment.jposition will be set to experiment.position if not set)
    //
    // additionally, the following global variables have to be set:
    //
    // %precission=0.001    (lower limit of evaluated matrix elements)
    // 
    // 'out' is a matrix with [a,a] elements (a=number of eigenstates)
    // and the following meaning:
    // Rate=out[final-state,initial-state]
    
    spec_length=experiment.spec_length;
       
    ve=experiment.Eigenval;
    ve=ve-min(real(ve));
    
    T=abs(experiment.T);
    lt=experiment.lt;
    w=experiment.atom(experiment.jposition).w;
    J=experiment.atom(experiment.jposition).J;
    if length(J)<>3 then
        J=[J J J];
    end

    a=size(experiment.Eigenvec,2);
    
    xrange=max([2*ve+20*T;20*T]);
    xpoints=floor(max([2*xrange/T;spec_length])); 
    x=linspace(-xrange,xrange,xpoints); //integration grid 
    
    f0=fstep_simple(x/T);
    
    tempexp=experiment;
    tempexp.ptip=tempexp.psample;
    tempexp.atom(:).U=0;
    tempexp.use_T_ts=%T;    //use it here! 
    tempexp.T_ts=J;
       
    
    out=zeros(a,a);
    for in=1:a,
        [rate,rater,raten,ratenr]=M35(tempexp,in);
    
        for mid=1:a
            if sum(abs(rate(mid,:))+ abs(rater(mid,:)))>%precission^2*sum(J)^2 then
                
                tlog1=ln_t2(ve(mid)-ve(in)+x,w,T,lt);
                tlog2=tlog1($:-1:1); //reverse the order
                
                for fin=1:a,
                    ep=fstep_simple((ve(fin)-ve(in)-x)/T);

                    ytemp=sum(-2*real(rate(mid,fin)*ep'.*tlog2.*f0'));
                    ytempr=sum(-2*real(rater(mid,fin)*ep'.*tlog1.*f0'));

                    out(fin,in)=out(fin,in)+(ytemp+ytempr)*2*xrange/(xpoints-1);
                                            
                end
            end
        end
    end
    
endfunction

function [out]=spec3rate_tt(experiment)
    // function to calculate the tip-tip transission probabilities 
    // in 3rd order 
    // experiment has to contain at least the following informations:
    //
    // experiment.T         (temperature)
    // experiment.lt        (lifetime broadening)
    // experiment.ptip      (tip spin-polarizaion)
    // experiment.psample   (sample Spin-polarization)
    // experiment.atom      (either the parameters of an atom or a list of atoms)
    // experiment.Eigenvec  (at least two eigenvectors of the Hamiltonian)
    // experiment.Eigenval  (at least two eigenvalues of the Hamiltonian)
    // experiment.position  (the position of the tip when we have a list of atoms)
    // experiment.jposition (the position where the sample-sample transition takes place)
    //                      (experiment.jposition will be set to experiment.position if not set)
    //
    // additionally, the following global variables have to be set:
    //
    // %precission=0.001    (lower limit of evaluated matrix elements)
    // 
    // 'out' is a matrix with [a,a] elements (a=number of eigenstates)
    // and the following meaning:
    // Rate=out[final-state,initial-state]
    
    spec_length=experiment.spec_length;
        
    ve=experiment.Eigenval;
    ve=ve-min(real(ve));
    
    T=abs(experiment.T);
    lt=experiment.lt;
    w=experiment.atom(experiment.jposition).w;
    J=-sqrt(experiment.A); // unfortunatelly only AFM coupling
 
    a=size(experiment.Eigenvec,2);
    
    xrange=max([2*ve+20*T;20*T]);
    xpoints=floor(max([xrange/T;spec_length])); 
    x=linspace(-xrange,xrange,xpoints); //integration grid 
    
    f0=fstep_simple(x/T);
    
    tempexp=experiment;
    tempexp.psample=tempexp.ptip;
    tempexp.atom(:).U=0;
    tempexp.use_T_ts=%T;    //use it here! 
    tempexp.T_ts=[J J J];
       
    
    out=zeros(a,a);
    for in=1:a,
        [rate,rater,raten,ratenr]=M35(tempexp,in);
    
        for mid=1:a
            if sum(abs(rate(mid,:))+ abs(rater(mid,:)))>%precission^2*sum(J)^2 then
                
                tlog1=ln_t2(ve(mid)-ve(in)+x,w,T,lt);
                tlog2=tlog1($:-1:1); //reverse the order
                
                for fin=1:a,
                    ep=fstep_simple((ve(fin)-ve(in)-x)/T);

                    ytemp=sum(-2*real(rate(mid,fin)*ep'.*tlog2.*f0'));
                    ytempr=sum(-2*real(rater(mid,fin)*ep'.*tlog1.*f0'));

                    out(fin,in)=out(fin,in)+(ytemp+ytempr)*2*xrange/(xpoints-1);
                                            
                end
            end
        end
    end

endfunction

////////////////////////////////////////////////////////////////////////////////
//some special graphical functions 
////////////////////////////////////////////////////////////////////////////////

function [bb,w]=Plotstates(S,B)
  // Plot the energy of the eigenstates in dependece of the field B=[0,(Bx, By, Bz)]
  // S must be the spin system with at least
  // S.S as the spin
  // S.g as the gyromagnetic factor
  //
  // if %zeroadj=%T then the lowest state is always at zero energy
  // output bb contains the energies at field |B|=w

  global %zeroadj;  
  
  b=linspace(0,1,1000);
  bb=b*sqrt(B(1)^2+B(2)^2+B(3)^2);
  clear w;
  
  if %zeroadj==%T then
    for i=1:1000, 
          w(:,i)=real(spec(HZ(S,b(i)*B)+Haniso(S)));    
          //we have to use "real" because sometimes the finite numerical 
          //precision prohibits the matrix of beeing symmetrical
          w(:,i)=w(:,i)-min(w(:,i));
    end;
  else
    for i=1:1000, 
          w(:,i)=real(spec(HZ(S,b(i)*B)+Haniso(S)));
          //we have to use "real" because sometimes the finite numerical 
          //precision prohibits the matrix of beeing symmetrical
    end;
  end;
  w=gsort(w,'r','i');
  plot(bb',w');
endfunction

function blochsphere(rho,varargin)
    // plots the blochsphere of the density matrix rho of a S=1/2 system
    // varargin(1) is the optional color   
    // if varargin(2)  is not empty, then labels are written
    // with the additional varargin(3) the labels can be provided ["N","S","E","W"]
    
    //some checks
  handle=gcf(); //get the actual graphic handle  
  drawlater();
  [lhs,rhs]=argn(0); 
  if rhs < 2 then
       col='blue' //default color
  else
       col=varargin(1);
  end
  
  if type(col)==10 then
     col=name2rgb(col);      //col is string
  end

    //check if rho is state-vector or density matrix
    if or(size(rho)==1) then
       rho=rho*rho'; 
    end;

    
    e.S=0.5*(size(rho,2)-1); // determine the spin of the rho
    
    //get the expectation values for Sx,Sy, and Sz.
    //then divide by S so that the values are between -1 and +1.
    
    r=[trace(Sx(e)*rho),trace(Sy(e)*rho),trace(Sz(e)*rho)]/e.S;
    
    if abs(r)==0 then
        r=[0, 0, 0.00001]; //helps to prevent 1/0.
    end
    
    //calculate variables needed for drawing the sphere
    
    gr=50; //gridpoints
    u = linspace(-%pi/2,%pi/2,gr);
    v = linspace(0,2*%pi,gr);
    x= cos(u)'*cos(v);
    xx=cos(v);
    yy=sin(v);
    y= cos(u)'*sin(v);
    z= sin(u)'*ones(v);
    
    scf(handle);
    param3d1([0;r(1)],[0;r(2)],[0;r(3)])
    
    e=gce()
    e.foreground=color(col(1),col(2),col(3));
    e.thickness=4;
    
    
    //create vector shape
    c1=rand(1,3); //random vector
    temp=find(r<>0); 
    
    temp=temp(1); //reduce to one element
    temp2=find([1 2 3] <> temp) //find the other elements
    c1(temp)=-sum(c1(temp2).*r(temp2))/r(temp);
    c1=c1/sqrt(sum(c1.^2));
    c2=crossp(c1,r)';
    c2=c2/sqrt(sum(c2.^2));
    
    for i=1:20,
        d=2*%pi/20*i;
        vec=r-0.1*r/sqrt(sum(r.^2))+0.05*sin(d)*c1+0.05*cos(d)*c2;
        vecx(2*i)=vec(1);
        vecx(2*i-1)=r(1);
        vecy(2*i)=vec(2);
        vecy(2*i-1)=r(2);
        vecz(2*i)=vec(3);
        vecz(2*i-1)=r(3);
    end
    
    scf(handle);
    param3d(vecx,vecy,vecz);
    e=gce();
    e.foreground=color(col(1),col(2),col(3));
    e.thickness=4;
    
    scf(handle);
    if rhs<3 then   //standart axes
        param3d1([-1;1],[0;0],[0;0]);
        e=gce();
        e.line_style=2;
        param3d1([0;0],[-1;1],[-0;0]);
        e=gce();
        e.line_style=2;
        param3d1([0;0],[0;0],[-1;1]);  
        e=gce();
        e.line_style=2;
    else         //longer and no tics
        param3d1([-1.1;1.1],[0;0],[0;0]);
        e=gce();
        e.line_style=2;
        param3d1([0;0],[-1.1;1.1],[-0;0]);
        e=gce();
        e.line_style=2;
        param3d1([0;0],[0;0],[-1.1;1.1]);  
        e=gce();
        e.line_style=2;
    end

    param3d1([r(1),r(1)],[r(2),r(2)],[0,r(3)]);
    //e=gce();
    //e.foreground=color(0.5,0.5,0.5);
    param3d1([r(1),r(1)],[0,r(2)],[0,0]);
    //e=gce();
    //e.foreground=color(0.5,0.5,0.5);
    param3d1([0,r(1)],[r(2),r(2)],[0,0]);
    //e=gce();
    //e.foreground=color(0.5,0.5,0.5);
    
    
    //draw the sphere
    scf(handle);
    plot3d3(xx,yy,zeros(gr,1),flags=[2,2,0,0]);
    plot3d3(zeros(gr,1),xx,yy,flags=[2,2,0,0]);
    if rhs>2 then
        legend=" @ @ ";
    else
        legend="$x$@$y$@$z$";
    end
    plot3d3(xx,zeros(gr,1),yy,theta=100,alpha=70,leg=legend,flags=[2,2,0,0]);
    h=gcf();
    hh=h.children;
    hh.isoview = "on";
    if rhs>2 then
        e=gca();
        e.axes_visible=["off","off","off"];
    end
    
    //labels
    if rhs>2 then   //make label
        if rhs==3 then  //use interal labels "up" and "down"
            label=["$|\!\uparrow\rangle$", "$|\!\downarrow\rangle$",..
            "$\frac{|\!\uparrow\rangle-|\!\downarrow\rangle}{\sqrt{2}}$",..
            "$\frac{|\!\uparrow\rangle+|\!\downarrow\rangle}{\sqrt{2}$"];
        else
            label=varargin(3);
        end
        
        xstring(1.1,0,label(1));
        h=gca();
        h.children(1).data=[0,0,1.2];
        h.children(1).text_box_mode = "centered";
        xstring(-1.1,0,label(2));
        h.children(1).data=[0,0,-1.2];
        h.children(1).text_box_mode = "centered";
        if prod(size(label))==4 then
            xstring(-1.1,0,label(3));
            h.children(1).data=[-1.7,0,0];
            h.children(1).text_box_mode = "centered";
            xstring(-1.1,0,label(4));
            h.children(1).data=[1.7,0,0];
            h.children(1).text_box_mode = "centered";
        end
        
        
    end
//    h.children(1).data=[0,0,1.1];
    drawnow();

endfunction


function [ev,vv]=drawstatesphere(rho)
    // This function plots the statesphere of the statevector or density matrix 
    // rho. It plots an ellipsoid with [<Sx>,<Sy>,<Sz>] as the center and radii
    // <delta Sx>, <delta Sy>, <delta Sz>
    
    // Recently I found a paper from Bharath, et al. (arXiv1801.00586)
    // where they use this (full) description for S>0.5 systems to discuss
    // geometric phases in quantum systems!
    
    
    //check if rho is state-vector or density matrix
    if or(size(rho)==1) then
       rho=rho*rho'; 
    end;
    
    //normalize rho
    l=trace(rho);
    if clean(l)==0 then
        error("Density matrix is not well defined!");
        abort;
    end
    
    rho=rho./l;
    S.S=(size(rho,1)-1)/2; //determine spin
    
    ev(1)=trace(Sx(S)*rho);   //calculate expection values
    ev(2)=trace(Sy(S)*rho);
    ev(3)=trace(Sz(S)*rho);
    
    ev=real(ev); //remove spurious imaginary parts 
    
    vv(1)=trace(Sx(S)^2*rho)-ev(1)^2; //calc. variance of evx
    vv(2)=trace(Sy(S)^2*rho)-ev(2)^2;
    vv(3)=trace(Sz(S)^2*rho)-ev(3)^2;
    
    vv=real(vv); //remove spurious imaginary parts 
    
    f=gcf();
    drawlater()
    f.color_map = coolcolormap(128);
    //f.color_map = hotcolormap(128);
    r=sqrt(abs(0.5*vv));orig=real(ev);
    
    deff("[x,y,z]=sph(alp,tet)",["x=r(1)*cos(alp).*cos(tet)+orig(1)*ones(tet)";..
     "y=r(2)*cos(alp).*sin(tet)+orig(2)*ones(tet)";..
     "z=r(3)*sin(alp)+orig(3)*ones(tet)"]);
    [xx,yy,zz]=eval3dp(sph,linspace(-%pi/2,%pi/2,40),linspace(0,%pi*2,20));
    [xx1,yy1,zz1]=eval3dp(sph,linspace(-%pi/2,%pi/2,40),linspace(0,%pi*2,20));
    cc=(xx+zz+2)*32;cc1=(xx1-orig(1)+zz1/max(r)+2)*32;
    plot3d1([xx xx1],[yy yy1],list([zz,zz1],[cc cc1]),theta=70,alpha=80,flag=[5,6,3]);
    
    //plot the sphere
    gr=100; //gridpoints
    r=sqrt(S.S*(S.S+1));
    v = linspace(0,2*%pi,gr);
    xx=r*cos(v);
    yy=r*sin(v);
    
    param3d1([-r;r],[0;0],[0;0]);
    e=gce();
    e.line_style=2;
    param3d1([0;0],[-r;r],[-0;0]);
    e=gce();
    e.line_style=2;
    param3d1([0;0],[0;0],[-r;r]);
    e=gce();
    e.line_style=2;
    
    legend=["$\langle m_x \rangle $@$\langle m_y \rangle $@$\langle m_z \rangle $"]
    
    plot3d3(xx,yy,zeros(gr,1),theta=70,alpha=80,flag=[6,6,3]);
    plot3d3(zeros(gr,1),xx,yy,theta=70,alpha=80,flag=[6,6,3]);
    plot3d3(xx,zeros(gr,1),yy,theta=100,alpha=70,leg=legend,flag=[6,6,4],);
    hh=f.children;
    hh.isoview = "on";
    hh.tight_limits = "on";
    drawnow();
    
endfunction

function drawstatehist2(rho,varargin)
    //this function plots the distibution of the statevector or density matrix rho
    //The diagonal elements are drawn in red-yellow
    //The conherences (off-diagonal) elements are drawn in blue
    //As varargin 'real' can be given, then only the real part will be drawn
    //As varargin 'scale',[min,max] can be given, then the scale is [min,max];
    //As varargin 'label','text' can be given, then "text" will be used as labels.
    //
    //"rho" has to be the densitymatrix
    //"varargin" cn be
    // (1) either "real", "full" or "auto" for the display of the matrix elements
    // (2) "scale", min, max for the scaling
    // (3) "label", vector of labelstrings
    //
    
    
    [lhs,rhs]=argn(0); global testt;
    
    if ~isdef("rho","l") then
        error(["Wrong number of input arguments",..
        "A desity matrix or state vector has to be given."]);
    end
    
    sc=%F;
    ticks_label="";
    if rhs > 1 then
        if varargin(1)=='label' then
            ticks_label=varargin(2);
        end
    end
    
     //check if rho is state-vector or density matrix
    if or(size(rho)==1) then
       rho=rho*rho'; 
    end;
    
    //normalize rho
    l=trace(rho);
    if clean(l-1,1e-4)>0 then
        error("Density matrix is not well defined! Sum of diagonal values >1");
        abort;
    end
    
    S=(size(rho,1)-1)/2; //determine the spin (if single spin)
    if ticks_label=="" then
        ticks_label=string([S:-1:-S]');
        ticks_label="$|"+ticks_label+"\rangle$";
    else
        ticks_label=matrix(ticks_label,size(rho,1),1);
    end
    
    ticks_position=[0.5:1:2*S+0.5]';
    ticks=tlist(["ticks","location","labels"],ticks_position, ticks_label);
    
    
    line=[0:0.2:size(rho,1)];
    line2=[line;line];
    line=0*line+size(rho,1);
    line=[0*line; line];
    
    rhodiag=eye(rho);
    diagmask=ones(rho);
    diagmask(find(rhodiag==0))=%nan;
    coherencemask=ones(rho);
    coherencemask(find(rhodiag==1))=%nan;
    
    drawlater();
    rho=full(rho);
    if re=="real" then  //if no imag parts are in rho
        bar3d(real(coherencemask.*rho));
        bar3d(real(diagmask.*rho));
    else                                 //if imag parts are in rho
        subplot(121);
        bar3d(real(coherencemask.*rho));
        subplot(122);
        bar3d(imag(coherencemask.*rho));
        param3d1(line2,line,0*line);
        param3d1(line,line2,0*line);    
        a=gca();
        a.cube_scaling='on';
        a.x_ticks=ticks;
        a.y_ticks=ticks;
        a.x_label.text="";
        a.y_label.text="";
        a.z_label.text="";
        a.z_label.text="$\Im[\chi]$"
        a.rotation_angles=[70,20];
        if sc then
            a.data_bounds(6)=sc_max;
            a.data_bounds(5)=sc_min;
        end
        subplot(121);
        bar3d(real(diagmask.*rho));
    end;
    a=gca();
    a.cube_scaling='on';
    a.children(1).hiddencolor=21;
    a.children(1).color_mode=5;
    
    a.x_ticks=ticks;
    a.y_ticks=ticks;
    a.x_label.text="";
    a.y_label.text="";
    a.z_label.text="";
    a.z_label.text="$\Re[\chi]$";
    
    if sc then
        a.data_bounds(6)=sc_max;
        a.data_bounds(5)=sc_min;
    end
            
    param3d1(line2,line,0*line);
    param3d1(line,line2,0*line);    
    
    a.rotation_angles=[70,20];
    drawnow();
        
endfunction

function drawstatehist(rho,varargin)
    //this function plots the distibution of the statevector or density matrix rho
    //The diagonal elements are drawn in red-yellow
    //The conherences (off-diagonal) elements are drawn in blue
    //As varargin 'real' can be given, then only the real part will be drawn
    //As varargin 'scale',[min,max] can be given, then the scale is [min,max];
    //As varargin 'label','text' can be given, then "text" will be used as labels.
    //
    //"rho" has to be the densitymatrix
    //"varargin" cn be
    // (1) either "real", "full" or "auto" for the display of the matrix elements
    // (2) "scale", min, max for the scaling
    // (3) "label", vector of labelstrings
    //
    
    
    [lhs,rhs]=argn(0); global testt;
    
    if ~isdef("rho","l") then
        error(["Wrong number of input arguments",..
        "A desity matrix or state vector has to be given."]);
    end
    
    re="auto";
    sc=%F;
    ticks_label="";
    if rhs > 1 then
        if varargin(1)=="real" then
            re="real";
            varargin(1)= null();
            rhs=rhs-1;
        elseif varargin(1)=="full" then
            re="full";
            varargin(1)= null();
            rhs=rhs-1;
        elseif varargin(1)=="auto" then
            re="auto";
            varargin(1)= null();
            rhs=rhs-1;
        end
    end
    if rhs > 1 then
        if varargin(1)=='scale' then
            sc=%T;
            sc_min=varargin(2);
            sc_max=varargin(3);
            varargin(1)= null();
            varargin(1)= null();
            varargin(1)= null();
            rhs=rhs-3;
        end
    end
    if rhs > 1 then
        if varargin(1)=='label' then
            ticks_label=varargin(2);
        end
    end
    
     //check if rho is state-vector or density matrix
    if or(size(rho)==1) then
       rho=rho*rho'; 
    end;
    
    //normalize rho
    l=trace(rho);
    if clean(l-1,1e-4)>0 then
        error("Density matrix is not well defined! Sum of diagonal values >1");
        abort;
    end
    
    S=(size(rho,1)-1)/2; //determine the spin (if single spin)
    if ticks_label=="" then
        ticks_label=string([S:-1:-S]');
        ticks_label="$|"+ticks_label+"\rangle$";
    else
        ticks_label=matrix(ticks_label,size(rho,1),1);
    end
    
    ticks_position=[0.5:1:2*S+0.5]';
    ticks=tlist(["ticks","location","labels"],ticks_position, ticks_label);
    
    
    line=[0:0.2:size(rho,1)];
    line2=[line;line];
    line=0*line+size(rho,1);
    line=[0*line; line];
    
    rhodiag=eye(rho);
    diagmask=ones(rho);
    diagmask(find(rhodiag==0))=%nan;
    coherencemask=ones(rho);
    coherencemask(find(rhodiag==1))=%nan;
    
   if re=="auto" then 
       if and (clean(imag(rho))==0) then
          re= "real"; 
      end
   end
  
    
    drawlater();
    rho=full(rho);
    if re=="real" then  //if no imag parts are in rho
        bar3d(real(coherencemask.*rho));
        bar3d(real(diagmask.*rho));
    else                                 //if imag parts are in rho
        subplot(121);
        bar3d(real(coherencemask.*rho));
        subplot(122);
        bar3d(imag(coherencemask.*rho));
        param3d1(line2,line,0*line);
        param3d1(line,line2,0*line);    
        a=gca();
        a.cube_scaling='on';
        a.x_ticks=ticks;
        a.y_ticks=ticks;
        a.x_label.text="";
        a.y_label.text="";
        a.z_label.text="";
        a.z_label.text="$\Im[\chi]$"
        a.rotation_angles=[70,20];
        if sc then
            a.data_bounds(6)=sc_max;
            a.data_bounds(5)=sc_min;
        end
        subplot(121);
        bar3d(real(diagmask.*rho));
    end;
    a=gca();
    a.cube_scaling='on';
    a.children(1).hiddencolor=21;
    a.children(1).color_mode=5;
    
    a.x_ticks=ticks;
    a.y_ticks=ticks;
    a.x_label.text="";
    a.y_label.text="";
    a.z_label.text="";
    a.z_label.text="$\Re[\chi]$";
    
    if sc then
        a.data_bounds(6)=sc_max;
        a.data_bounds(5)=sc_min;
    end
            
    param3d1(line2,line,0*line);
    param3d1(line,line2,0*line);    
    
    a.rotation_angles=[70,20];
    drawnow();
        
endfunction



function drawanisotropy(H,n,varargin)
    //this function plots the energy it costs to rotate the density 
    //matrix or statevector n into different directions
    //
    //H is the Hamiltonian of the system which contains exactly one spin.
    //We can get H for example by H=Haniso(S)+HZ(S,B) having S.S, S.g, S.D, 
    //S.E, S.DD, and S.G defined. Other creations using the Stevens operators
    //are also possible.
    //Depending on the dimension of n it is either: 
    //The state number in the base system (n is integer)
    //The eigenstate (n is vector)
    //The density matrix (n is matrix)
    //As varargin a text can be given wich is printed
    
    [lhs,rhs]=argn(0); 
    if rhs > 2 then 
        txt=varargin(1); 
    else
        txt="$\mbox{Energy of the state vector when pointing in arb. directions}$"
    end
    
    S.S=(size(H,1)-1)/2; //determine the spin
    
    if size(n)==[1,1] then
        dmz=zeros(H); dmz(n,n)=1; //set the desity matrix
    elseif size(n)==[size(H,1),1] then
        dmz=n*n';
    else
        dmz=n;
    end

    phi=linspace(0,2*%pi,60);
    tet=linspace(0,%pi,30);
    
    tet2=ones(length(phi),1).*.tet;
    phi2=phi'.*.ones(1,length(tet));

    x=sin(tet2).*cos(phi2);
    y=sin(tet2).*sin(phi2);
    z=cos(tet2);
    
    e=zeros(length(phi),length(tet));
    for i=1:length(tet),
        for j=1:length(phi),
            rotm=Srot(S,tet(i),[0 1 0]);
            dm=rotm'*dmz*rotm;
            rotm=Srot(S,phi(j),[0 0 1]);
            dm=rotm'*dm*rotm;
            e(j,i)=trace(H*dm);
        end
    end
   
    e=real(e)-min(real(e));
    if and(clean(e)==0) then
        e=e+1;
    end
    x=x.*e;
    y=y.*e;
    z=z.*e;
    
    inda=ones(1,length(phi)-1).*.[0 1 length(phi)+1 length(phi)]+ (1:length(phi)-1).*.[1 1 1 1];
    ind2a=ones(1,length(tet)-1).*.inda+((0:length(tet)-2)*length(phi)).*.ones(inda);
    
    nx=prod(size(ind2a));
    vx=matrix(x(ind2a),4,nx/4);
    vy=matrix(y(ind2a),4,nx/4);
    vz=matrix(z(ind2a),4,nx/4);
    
    cc=real(sqrt(vx.^2+vy.^2+vz.^2));cc=cc/max(cc)*128+0;
    //cc=vx+vz;cc=cc/max(cc)*128+64;
    drawlater();    
    plot3d1([vx],[vy],list([vz],[-cc]),theta=70,alpha=80,flag=[5,6,3]);
   
    //plot the sphere
    gr=100; //gridpoints
    r=max(sqrt(x.^2+y.^2+z.^2)); //r=5.5
    v = linspace(0,2*%pi,gr);
    xx=r*cos(v);
    yy=r*sin(v);
    
    param3d1([-r;r],[0;0],[0;0]);
    eg=gce();
    eg.line_style=2;
    param3d1([0;0],[-r;r],[-0;0]);
    eg=gce();
    eg.line_style=2;
    param3d1([0;0],[0;0],[-r;r]);
    eg=gce();
    eg.line_style=2;
    
    plot3d3(xx,yy,zeros(gr,1),theta=70,alpha=80,flag=[6,6,3]);
    plot3d3(zeros(gr,1),xx,yy,theta=70,alpha=80,flag=[6,6,3]);
    plot3d3(xx,zeros(gr,1),yy,theta=100,alpha=70,flag=[6,6,4]);
    f=gcf();
    //f.children(1).axes_bounds=[0 0 1 0.6];
    f.children(1).isoview = "on";
    f.children(1).tight_limits = "on";
    f.color_map = jetcolormap(128);
    colorbar(min(e),max(e));
   // xset("clipoff")
   
   a=-0.188;
   Rx=[1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
   b=-0.314;
   Ry=[cos(b) 0 sin(b); 0 1 0; -sin(b) 0 cos(b)];
   c=-0.38;
   Rz=[cos(c) -sin(c) 0; sin(c) cos(c) 0; 0 0 1];
//   v=Rz*Ry*Rx*[0; 0; -1];
   v=Rx*Ry*Rz*[0; 0; 1];
   w=Rx*Ry*Rz*[0; -1; 0];
   //r=r*1.5;
   //xarrows([0 0;-w(1)*r -v(1)*r],[0 0; -w(2)*r -v(2)*r], [0 0 ;-w(3)*r -v(3)*r]);

    drawnow();

    title(txt,'fontsize',2);   

endfunction   

function drawenergy(experiment,n)
    //this function plots the energy it costs to rotate the density 
    //matrix n into different directions.
    //
    //experiment contains all information of the system. In particular:
    //
    //experiment.atom   (either the parameters of an atom or a list of atoms)
    //experiment.B      (the extermal magnetic field)
    //
    //Depending on the dimension of n it is either: 
    //The state number in the base system (n is integer)
    //The eigenstate (n is vector)
    //The density matrix (n is matrix)
    
    //if ~exists('%stopcalc') then
        %stopcalc=%F;
    //end
    
    atomlink=experiment.atom;
    nbr=size(atomlink,2);
    sz=1;
    for i=1:nbr,
            sz=sz*(2*atomlink.S(i)+1);
    end
    
    if size(n)==[1,1] then
        dmz=zeros(sz,sz); dmz(n,n)=1; //set the desity matrix
    elseif size(n)==[sz,1] then
        dmz=n*n';
    else
        dmz=n;
    end
    
    H=hamiltonian(experiment);
    
    tet=linspace(0,2*%pi,60);
    phi=linspace(0,%pi,30);
    
    tet2=ones(length(phi),1).*.tet;
    phi2=phi'.*.ones(1,length(tet));

    x=sin(tet2).*cos(phi2);
    y=sin(tet2).*sin(phi2);
    z=cos(tet2);
    
    e=zeros(length(phi),length(tet));;
    winH=waitbar('Calcuating Energies');
    for i=1:length(tet),
        if ~%stopcalc then
            waitbar(i/length(tet),winH);
        else
            close(winH);
            return;
        end;
        for j=1:length(phi),
            rotm=Srotn(atomlink,tet(i),[0 1 0]);
            dm=rotm'*dmz*rotm;
            rotm=Srotn(atomlink,phi(j),[0 0 1]);
            dm=rotm'*dm*rotm;
            e(j,i)=trace(H*dm); 
        end
    end
    close(winH);
   
    e=real(e)-min(real(e));
    if and(clean(e)==0) then
        e=e+1;
    end
    x=x.*e;
    y=y.*e;
    z=z.*e;
    
    inda=ones(1,length(phi)-1).*.[0 1 length(phi)+1 length(phi)]+ (1:length(phi)-1).*.[1 1 1 1];
    indb=ones(1,length(phi)-1).*.[1 0 length(phi) length(phi)+1]+ (1:length(phi)-1).*.[1 1 1 1];
    ind2a=ones(1,length(tet)-1).*.inda+((0:length(tet)-2)*length(phi)).*.ones(inda);
    ind2b=ones(1,length(tet)-1).*.indb+((0:length(tet)-2)*length(phi)).*.ones(indb);
    
    ind2=[ind2a(1:length(ind2a)/2),ind2b(length(ind2b)/2+1:$)]
    nx=prod(size(ind2));
    vx=matrix(x(ind2),4,nx/4);
    vy=matrix(y(ind2),4,nx/4);
    vz=matrix(z(ind2),4,nx/4);
    
    cc=real(sqrt(vx.^2+vy.^2+vz.^2));cc=cc/max(cc)*128+0;
    //cc=vx+vz;cc=cc/max(cc)*128+64;
    drawlater();    
    plot3d1([vx],[vy],list([vz],[cc]),theta=70,alpha=80,flag=[5,6,3]);
   
    //plot the sphere
    gr=100; //gridpoints
    r=max(sqrt(x.^2+y.^2+z.^2));
    v = linspace(0,2*%pi,gr);
    xx=r*cos(v);
    yy=r*sin(v);
    
    param3d1([-r;r],[0;0],[0;0]);
    eg=gce();
    eg.line_style=2;
    param3d1([0;0],[-r;r],[-0;0]);
    eg=gce();
    eg.line_style=2;
    param3d1([0;0],[0;0],[-r;r]);
    eg=gce();
    eg.line_style=2;
    
    plot3d3(xx,yy,zeros(gr,1),theta=70,alpha=80,flag=[6,6,3]);
    plot3d3(zeros(gr,1),xx,yy,theta=70,alpha=80,flag=[6,6,3]);
    plot3d3(xx,zeros(gr,1),yy,theta=100,alpha=70,flag=[6,6,4]);
    f=gcf();
    //f.children(1).axes_bounds=[0 0 1 0.6];
    f.children(1).isoview = "on";
    f.children(1).tight_limits = "on";
    f.color_map = jetcolormap(128);
    colorbar(min(e),max(e));
    
    drawnow();
//    title("$\mbox{Visualization of the Magnetic Anisotropy [meV]}$")
    title("$\mbox{Energy of the state vector when pointing in arb. directions}$")

endfunction   

function [dm]=densitymatrix(experiment)
    //calculate the densitymatrix from the eigenstates, eigenenergies and
    //temperature defined in "experiment".
    
    [a,b]=size(experiment.Eigenvec);
    occ=Occupation(experiment.Eigenval,experiment.T);
    dm=spzeros(a,a);
    for i=1:b,
        if occ(i)>%precission^2 then
            dm=dm+occ(i)*sparse(experiment.Eigenvec(:,i))*sparse(experiment.Eigenvec(:,i))';
        end
    end
    
endfunction

////////////////////////////////////////////////////////////////////////////////
//some QCorrelation functions
////////////////////////////////////////////////////////////////////////////////

function calculate_ptipK(experiment)
    
    global experiment;
    b=size(experiment.Eigenvec,2);
    
    experiment.ptipK=zeros(b,3);
    experiment.psampleK=zeros(b,3);
    
    for i=1:b,      //loop over all eigenstates
        
        if experiment.sef<>0 then  //only do the huge work if necessary.

            //create density matrix dm=|b><b|
            dm=experiment.Eigenvec(:,i)*experiment.Eigenvec(:,i)';
            
            if experiment.corr_tip then  
            
                //reduce to the spin on the tip by tracing out all others
                dm_tip=traceout(dm,experiment.atom,1);
                     
                et.S=0.5*(size(dm_tip,2)-1); // determine the spin of the tip
              
                //get the expectation values for Sx,Sy, and Sz.
                //then divide by S so that the values are between -1 and +1.
                rt=[trace(Sx(et)*dm_tip),trace(Sy(et)*dm_tip),trace(Sz(et)*dm_tip)]/et.S;
                rt=real(rt);    //remove imag parts     
        
                //the polarization in the tip shall now be r times Jrho0 times the
                //scaling factor "sef". 
                //Thus, for Jrho0*sef<0 we get AFM coupling, 
                //while for Jrho0*sef>0 we get FM coupling
            
                rt=experiment.sef*(rt.*experiment.atom(1).J);
    
                //write this tip-polarization in a list for every eigenstate
                experiment.ptipK(i,:)=[rt(1), rt(2), rt(3)];
            else
                // use the spin polarization value from experiment.ptip
                experiment.ptipK(i,:)=experiment.ptip;
            end
      
            if experiment.corr_sample then  //do the same for the sample states
                //reduce to the spin on the sample by tracing out all others
                dm_sample=traceout(dm,experiment.atom,experiment.position);
                es.S=0.5*(size(dm_sample,2)-1); // determine the spin of the sample
                rs=[trace(Sx(es)*dm_sample),trace(Sy(es)*dm_sample),trace(Sz(es)*dm_sample)]/es.S;
                rs=real(rs);    //remove imag parts     
                rs=experiment.sef*(experiment.atom(experiment.position).J.*rs);
                experiment.psampleK(i,:)=[rs(1), rs(2), rs(3)];
            else
                // use the spin polarization value from experiment.psample
                experiment.psampleK(i,:)=experiment.psample;
            end;
            
        else   //if sef=0 then use the regular polarization in tip and sample
            experiment.ptipK(i,:)=experiment.ptip;
            experiment.psampleK(i,:)=experiment.psample;
        end    
    end //loop's end
    
endfunction

function [out]=Rate2nd2_func_tip(experiment)
    //version 1.6
    global calc experiment;
    [a,b]=size(experiment.Eigenvec);
    
    txt=getfield(1,experiment);
    if grep(txt,'use_T_ts')==[] then
        use_T_ts=%F
    else
        use_T_ts=experiment.use_T_ts;
    end
    
    calculate_ptipK(experiment);
    
    out=zeros(b,b); //initialize the output matrix;
    nr=experiment.position;
    E=experiment.Eigenvec;

    natoms=size(experiment.atom,2);
    
    ma_s_x=1;
    ma_s_y=1;
    ma_s_z=1;
    ma_s_u=1;

    mm=1;
    for i=1:(nr-1),
        mm=mm.*.sparse(S1(experiment.atom(i)));
    end
    
    ma_s_x=ma_s_x.*.mm;
    ma_s_y=ma_s_y.*.mm;
    ma_s_z=ma_s_z.*.mm;
    ma_s_u=ma_s_u.*.mm;
         
    S=experiment.atom(nr);

    if use_T_ts then
        J=experiment.T_ts;      //new (version 1.6)
    else
        J=[1 1 1];
    end

    sx=sparse(Sx(S));      
    sy=sparse(Sy(S));
    sz=sparse(Sz(S));
    su=S.U*sparse(S1(S));
      
    ma_s_x=J(1)*ma_s_x.*.sx;
    ma_s_y=J(2)*ma_s_y.*.sy;
    ma_s_z=J(3)*ma_s_z.*.sz;
    ma_s_u=ma_s_u.*.su;
    
    mm=1;
    for i=nr+1:natoms,
        mm=mm.*.sparse(S1(experiment.atom(i)));
    end

    ma_s_x=ma_s_x.*.mm;
    ma_s_y=ma_s_y.*.mm;
    ma_s_z=ma_s_z.*.mm;
    ma_s_u=ma_s_u.*.mm;

    ma_s_x=E'*ma_s_x*E;   
    ma_s_y=E'*ma_s_y*E;
    ma_s_z=E'*ma_s_z*E;
    ma_s_u=E'*ma_s_u*E;
    
    [x,y,z,u]=etransport_func_tip(experiment);

   out=zeros(b,b);    
   for i=1:b,          //loop over any initial state |i>
      for j=1:b,      //loop over any final state |j>    
          x1=ma_s_x(i,j)*x(:,:,i,j);
          y1=ma_s_y(i,j)*y(:,:,i,j);
          z1=ma_s_z(i,j)*z(:,:,i,j);
          u1=ma_s_u(i,j)*u(:,:,i,j);
        
          out(i,j)=sum(abs((x1+y1+z1+u1).^2));
      end,    //loop's end
   end         //loop's end
   

//   for i=1:1,
//       for j=1:1,
//           x1=ma_s_x.*matrix(x(i,j,:,:),b,b);
//           y1=ma_s_y.*matrix(y(i,j,:,:),b,b);
//           z1=ma_s_z.*matrix(z(i,j,:,:),b,b);
//           u1=ma_s_u.*matrix(u(i,j,:,:),b,b);
//           
//           out=out+abs(x1+y1+z1+u1).^2;
//       end
//   end
    
endfunction

function [out]=Rate2nd2_func_tip_2(experiment)
    //version 1.6
    //experiment.func_tip==2 
    global calc experiment;
    [a,b]=size(experiment.Eigenvec);
    
    txt=getfield(1,experiment);
    if grep(txt,'use_T_ts')==[] then
        use_T_ts=%F
    else
        use_T_ts=experiment.use_T_ts;
    end
    
    calculate_ptipK(experiment);
    
    out=zeros(b,b); //initialize the output matrix;
    nr=experiment.position;
    E=experiment.Eigenvec;

    natoms=size(experiment.atom,2);
    
    //include tip's spin at position=1 into calculation  because
    //it's a spectroscopically active, i.e. 'bright' spin
    S=experiment.atom(1);

    if use_T_ts then
        J=experiment.T_ts;      //new (version 1.6)
    else
        J=[1 1 1];
    end
  
    ma_t_x=J(1)*sparse(Sx(S));  // ma_t_ is the transition matrix for tunneling 
    ma_t_y=J(2)*sparse(Sy(S));  // having excitations only in the tip
    ma_t_z=J(3)*sparse(Sz(S));

    //the minus originates from having the spin on the tip (correct??)
    ma_t_u=-S.U*sparse(S1(S));
        
    ma_ts_x=ma_t_x;       // ma_ts_ is the transition matrix for tunneling 
    ma_ts_y=ma_t_y;       // having excitations simultaneously in tip and sample
    ma_ts_z=ma_t_z;
    ma_ts_u=ma_t_u;
        
    mm=1;
    for i=2:natoms,
        mm=mm.*.sparse(S1(experiment.atom(i)));
    end

    ma_t_x=ma_t_x.*.mm;     // ma_t_ is fully prepared now
    ma_t_y=ma_t_y.*.mm;
    ma_t_z=ma_t_z.*.mm;
    ma_t_u=ma_t_u.*.mm;

    ma_s_x=1;      // ma_s_ is the transition matrix for tunneling 
    ma_s_y=1;      // having excitations only in the sample
    ma_s_z=1;
    ma_s_u=1;

    mm=1;
    for i=1:(nr-1),
        mm=mm.*.sparse(S1(experiment.atom(i)));
    end
    
    ma_s_x=ma_s_x.*.mm;
    ma_s_y=ma_s_y.*.mm;
    ma_s_z=ma_s_z.*.mm;
    ma_s_u=ma_s_u.*.mm;
         
    S=experiment.atom(nr);

    if use_T_ts then
        J=experiment.T_ts;      //new (version 1.6)
    else
        J=[1 1 1];
    end

    sx=sparse(Sx(S));
    sy=sparse(Sy(S));
    sz=sparse(Sz(S));
    su=S.U*sparse(S1(S));
      
    ma_s_x=J(1)*ma_s_x.*.sx;
    ma_s_y=J(2)*ma_s_y.*.sy;
    ma_s_z=J(3)*ma_s_z.*.sz;
    ma_s_u=ma_s_u.*.su;
    
    mm=1;
    for i=2:(nr-1),
        mm=mm.*.sparse(S1(experiment.atom(i)));
    end
       
    ma_ts_x=ma_ts_x.*.mm;
    ma_ts_y=ma_ts_y.*.mm;
    ma_ts_z=ma_ts_z.*.mm;
    ma_ts_u=ma_ts_u.*.mm;
    
    ma_ts_x=ma_ts_x.*.sx;       //we use only one T_ts for x,y,z.
    ma_ts_y=ma_ts_y.*.sy;       //however, for u we multiply u_t*u_s...
    ma_ts_z=ma_ts_z.*.sz;
    ma_ts_u=ma_ts_u.*.su;
        
    mm=1;
    for i=nr+1:natoms,
        mm=mm.*.sparse(S1(experiment.atom(i)));
    end

    ma_s_x=ma_s_x.*.mm; //ma_s_ is fully prepared now
    ma_s_y=ma_s_y.*.mm;
    ma_s_z=ma_s_z.*.mm;
    ma_s_u=ma_s_u.*.mm;

    ma_ts_x=ma_ts_x.*.mm;   //ma_ts_ is fully prepared now
    ma_ts_y=ma_ts_y.*.mm;
    ma_ts_z=ma_ts_z.*.mm;
    ma_ts_u=ma_ts_u.*.mm;
        
    ma_ts_x=E'*ma_ts_x*E;   //<phi|ma_ts|phi> : calculate transition matrix elements
    ma_ts_y=E'*ma_ts_y*E;
    ma_ts_z=E'*ma_ts_z*E;
    ma_ts_u=E'*ma_ts_u*E;

    ma_t_x=E'*ma_t_x*E;   //<phi|ma_t|phi> : calculate transition matrix elements
    ma_t_y=E'*ma_t_y*E;
    ma_t_z=E'*ma_t_z*E;
    ma_t_u=E'*ma_t_u*E;
   
    ma_s_x=E'*ma_s_x*E;   //<phi|ma_s|phi> : calculate transition matrix elements
    ma_s_y=E'*ma_s_y*E;
    ma_s_z=E'*ma_s_z*E;
    ma_s_u=E'*ma_s_u*E;
    
    [x,y,z,u]=etransport_func_tip(experiment);
    
    U_t=2*experiment.atom(1).U;    //the factor 2 is here correct.
    U_s=2*experiment.atom(nr).U;

    [xx]=1*e2transport(experiment);
      
    for i=1:b,          //loop over any initial state |i>
      for j=1:b,      //loop over any final state |j>    
          x1=1*(U_t*ma_s_x(i,j)+U_s*ma_t_x(i,j))*x(:,:,i,j);
          y1=1*(U_t*ma_s_y(i,j)+U_s*ma_t_y(i,j))*y(:,:,i,j);
          z1=1*(U_t*ma_s_z(i,j)+U_s*ma_t_z(i,j))*z(:,:,i,j);
          u1=1*(U_t*ma_s_u(i,j)+U_s*ma_t_u(i,j))*u(:,:,i,j);
          
          x2=1*ma_ts_x(i,j)*xx(:,:,i,j);
          y2=1*ma_ts_y(i,j)*xx(:,:,i,j);
          z2=1*ma_ts_z(i,j)*xx(:,:,i,j);
          u2=1*ma_ts_u(i,j)*xx(:,:,i,j); //factor 2*2=4=?
              
          out(i,j)=sum(abs((x1+x2+y1+y2+z1+z2+u1+u2).^2));
      end
    end

endfunction

function [e,er,en,enr]=M35_func_tip_2(experiment,in)
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

    //first we calculate the transition matrices for the electrons ein 3rd order
    
   E=experiment.Eigenvec; a=size(E);
   
   jpos=experiment.jposition;   //backup jposition
   experiment.jposition=1;      //set to tip position

   //calculate scattering in tip
   [e_sig_t, er_sig_t, en_sig_t, enr_sig_t]=e_epsilon_func_tip(experiment,in);

   experiment.jposition=jpos;      //reset jposition
    
   //calculate scattering in sample
   [e_sig, er_sig, en_sig, enr_sig]=e_epsilon_func_tip(experiment,in);    

   txt=getfield(1,experiment);
   if grep(txt,'entanglement')==[] then
       experiment.entanglement=%F; //standart is no entanglement
   end;
   
   TJ=[1 1 1];      //standart value
   
   if grep(txt,'use_T_ts')==[] then
       experiment.use_T_ts=%f;
   elseif experiment.use_T_ts then
       TJ=experiment.T_ts;
   end

   nr=experiment.position;
   nrj=experiment.jposition;
   natoms=size(experiment.atom,2);
   
   //include tip's spin at position=1 into calculation  because
   //it's a spectroscopically active, i.e. 'bright' spin
   S_tip=experiment.atom(1);

   //ma_t_ is the matrix with excitations only in the tip
   ma_t_x=TJ(1)*sparse(Sx(S_tip));
   ma_t_y=TJ(2)*sparse(Sy(S_tip));
   ma_t_z=TJ(3)*sparse(Sz(S_tip));

   //the minus originates from having the spin on the tip 
   ma_t_u=-S_tip.U*sparse(S1(S_tip));
       
   //ma_ts_ is the matrix with excitations simultaneously in tip and sample       
   ma_ts_x=ma_t_x;
   ma_ts_y=ma_t_y;
   ma_ts_z=ma_t_z;
   ma_ts_u=ma_t_u;
        
   mm=1;
   for i=2:natoms,
       mm=mm.*.sparse(S1(experiment.atom(i)));
   end
   
   ma_t_x=ma_t_x.*.mm; //ma_t_ is fully prepared now
   ma_t_y=ma_t_y.*.mm;
   ma_t_z=ma_t_z.*.mm;
   ma_t_u=ma_t_u.*.mm;

   //ma_s_ is the matrix with excitations only in the sample
   ma_s_x=1;
   ma_s_y=1;
   ma_s_z=1;
   ma_s_u=1;

   mm=1;
   for i=1:(nr-1),
       mm=mm.*.sparse(S1(experiment.atom(i)));
   end
    
   ma_s_x=ma_s_x.*.mm;
   ma_s_y=ma_s_y.*.mm;
   ma_s_z=ma_s_z.*.mm;
   ma_s_u=ma_s_u.*.mm;
         
   S=experiment.atom(nr);

   sx=sparse(Sx(S));
   sy=sparse(Sy(S));
   sz=sparse(Sz(S));
   su=S.U*sparse(S1(S));
      
   ma_s_x=TJ(1)*ma_s_x.*.sx;
   ma_s_y=TJ(2)*ma_s_y.*.sy;
   ma_s_z=TJ(3)*ma_s_z.*.sz;
   ma_s_u=ma_s_u.*.su;
    
   mm=1;
   for i=2:(nr-1),
       mm=mm.*.sparse(S1(experiment.atom(i)));
   end
       
   ma_ts_x=ma_ts_x.*.mm;
   ma_ts_y=ma_ts_y.*.mm;
   ma_ts_z=ma_ts_z.*.mm;
   ma_ts_u=ma_ts_u.*.mm;
   
   ma_ts_x=ma_ts_x.*.sx;       //we use only one T_ts for x,y,z.
   ma_ts_y=ma_ts_y.*.sy;       //however, for u we multiply u_t*u_s...
   ma_ts_z=ma_ts_z.*.sz;
   ma_ts_u=ma_ts_u.*.su;
        
   mm=1;
   for i=nr+1:natoms,
       mm=mm.*.sparse(S1(experiment.atom(i)));
   end

   ma_s_x=ma_s_x.*.mm; //ma_s_ is fully prepared now
   ma_s_y=ma_s_y.*.mm;
   ma_s_z=ma_s_z.*.mm;
   ma_s_u=ma_s_u.*.mm;

   ma_ts_x=ma_ts_x.*.mm;   //ma_ts_ is fully prepared now
   ma_ts_y=ma_ts_y.*.mm;
   ma_ts_z=ma_ts_z.*.mm;
   ma_ts_u=ma_ts_u.*.mm;
        
   U_t=2*experiment.atom(1).U;
   U_s=2*experiment.atom(nr).U;
   
   //calculate the transiton matrix elements in the tip and sample spin system
   //i.e. the transtions <f|U_t*S_s+U_s*S_t|i> and <f|S_ts|i>
   matrxx=E'*(U_t*ma_s_x+U_s*ma_t_x)*E;
   matrxy=E'*(U_t*ma_s_y+U_s*ma_t_y)*E;
   matrxz=E'*(U_t*ma_s_z+U_s*ma_t_z)*E;
   matrxu=E'*(U_t*ma_s_u+U_s*ma_t_u)*E;
   matrxts=E'*(ma_ts_x+ma_ts_y+ma_ts_z)*E;
   matrxtsu=E'*ma_ts_u*E;

   S_tip=experiment.atom(1);      
      
   if length(S_tip.J) == 1 then
     J=S_tip.J;
     S_tip.J(1)=J;S_tip.J(2)=J;S_tip.J(3)=J;
   end

   Tmatrxx=S_tip.J(1)*sparse(Sx(S_tip));
   Tmatrxy=S_tip.J(2)*sparse(Sy(S_tip));
   Tmatrxz=S_tip.J(3)*sparse(Sz(S_tip));
 
   mm=1;
   for i=2:natoms,
       mm=mm.*.sparse(S1(experiment.atom(i)));
   end

   Tmatrxx=Tmatrxx.*.mm;
   Tmatrxy=Tmatrxy.*.mm;
   Tmatrxz=Tmatrxz.*.mm;
   
   Tmatrxx=E'*Tmatrxx*E;
   Tmatrxy=E'*Tmatrxy*E;
   Tmatrxz=E'*Tmatrxz*E;
   Tmatrxu=spzeros(Tmatrxx);
   
   if ~experiment.entanglement then    //no entanglement

      S=experiment.atom(nrj);      
      mm=1;            
      
      for i=1:nrj-1,
          mm=mm.*.sparse(S1(experiment.atom(i)));
      end
                          
      if length(S.J) == 1 then
         JT=S.J;
         S.J(1)=JT;S.J(2)=JT;S.J(3)=JT;
      end

      Smatrxx=S.J(1)*(mm.*.sparse(Sx(S)));
      Smatrxy=S.J(2)*(mm.*.sparse(Sy(S)));
      Smatrxz=S.J(3)*(mm.*.sparse(Sz(S)));
      
      for i=nrj+1:natoms,
          tempm=sparse(S1(experiment.atom(i)))
          Smatrxx=Smatrxx.*.tempm;
          Smatrxy=Smatrxy.*.tempm;
          Smatrxz=Smatrxz.*.tempm;
      end
      
       Smatrxx=E'*Smatrxx*E;
       Smatrxy=E'*Smatrxy*E;
       Smatrxz=E'*Smatrxz*E;

   else     //entanglement on sample site

      Smatrxx=spzeros(matrxx);
      Smatrxy=spzeros(matrxx);
      Smatrxz=spzeros(matrxx);
      
      for j=2:natoms,    //start at atom #2 because the tip is atom #1
          Stmatrxx=sparse(S1(experiment.atom(1)));                                
      
          for i=2:(j-1),
              Stmatrxx=Stmatrxx.*.sparse(S1(experiment.atom(i)));
          end
          
          Stmatrxy=Stmatrxx;
          Stmatrxz=Stmatrxx;
      
          Stmatrxx=Stmatrxx.*.sparse(Sx(experiment.atom(j)));
          Stmatrxy=Stmatrxy.*.sparse(Sy(experiment.atom(j)));
          Stmatrxz=Stmatrxz.*.sparse(Sz(experiment.atom(j)));
          
          for i=j+1:natoms,
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
  end         

  Smatrxu=spzeros(Smatrxx); 
  mone=ones(1,a(2));
  
  so1=list(matrxx,matrxy,matrxz,matrxu-matrxtsu-matrxts);
  so2=list(Smatrxx,Smatrxy,Smatrxz,Smatrxu);
  so2t=list(Tmatrxx,Tmatrxy,Tmatrxz,Tmatrxu);
//  so2=list(Smatrxx-Tmatrxx,Smatrxy-Tmatrxy,Smatrxz-Tmatrxz,Smatrxu);
  so3=list(matrxx,matrxy,matrxz,matrxu-matrxtsu+matrxts);
//  so3=list(matrxx,matrxy,matrxz,matrxu);
  e=zeros(matrxx);er=e;en=e;enr=e;
  for j=1:4,
    for k=1:4,
      for l=1:4,
        if or(e_sig(:,:,j,k,l)<>0) then
          e=e-1*e_sig(:,:,j,k,l).*(mone'.*.so3(l)(in,:)).*(so2(k)).'.*(mone.*.so1(j)(:,in));
        end
        if or(en_sig_t(:,:,j,k,l)<>0) then 
          e=e-1*en_sig_t(:,:,j,k,l).*(mone'.*.so1(l)(in,:)).*(so2t(k)).'.*(mone.*.so3(j)(:,in));
        end
        if or(er_sig(:,:,j,k,l)<>0) then 
          er=er-1*er_sig(:,:,j,k,l).*(mone'.*.so3(l)(in,:)).*(so1(k)).'.*(mone.*.so2(j)(:,in));
        end
        if or(enr_sig_t(:,:,j,k,l)<>0) then 
          er=er-1*enr_sig_t(:,:,j,k,l).*(mone'.*.so1(l)(in,:)).*(so3(k)).'.*(mone.*.so2t(j)(:,in));
        end
        if or(en_sig(:,:,j,k,l)<>0) then
          en=en-1*en_sig(:,:,j,k,l).*(mone'.*.so1(l)(in,:)).*(so2(k)).'.*(mone.*.so3(j)(:,in));
        end
        if or(e_sig_t(:,:,j,k,l)<>0) then
          en=en-1*e_sig_t(:,:,j,k,l).*(mone'.*.so3(l)(in,:)).*(so2t(k)).'.*(mone.*.so1(j)(:,in));
        end
        if or(enr_sig(:,:,j,k,l)<>0) then
          enr=enr-1*enr_sig(:,:,j,k,l).*(mone'.*.so1(l)(in,:)).*(so3(k)).'.*(mone.*.so2(j)(:,in));
        end
        if or(er_sig_t(:,:,j,k,l)<>0) then
          enr=enr-1*er_sig_t(:,:,j,k,l).*(mone'.*.so3(l)(in,:)).*(so1(k)).'.*(mone.*.so2t(j)(:,in));
        end
      end
    end
  end
   
//  e=clean(e);
//  er=clean(er);
//  en=clean(en);
//  enr=clean(enr);
    
endfunction


function [e,er,en,enr]=M35_func_tip(experiment,in)
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

    //first we calculate the transition matrices for the electrons ein 3rd order
    
   E=experiment.Eigenvec; a=size(E);
    
   [e_sig, er_sig, en_sig, enr_sig]=e_epsilon_func_tip(experiment,in);    
   
   txt=getfield(1,experiment);
   if grep(txt,'entanglement')==[] then
       experiment.entanglement=%F; //standart is no entanglement
   end;
   
   TJ=[1 1 1];  //standart value
   
   if grep(txt,'use_T_ts')==[] then
       experiment.use_T_ts=%f;
   elseif experiment.use_T_ts then
       TJ=experiment.T_ts;
   end


   nr=experiment.position;
   nrj=experiment.jposition;
   natoms=size(experiment.atom);
   
   matrxx=1;
   
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
   
   matrxx=TJ(1)*matrxx;    //new in Version 1.6 we scale the tip sample scattering
   matrxy=TJ(2)*matrxy;
   matrxz=TJ(3)*matrxz;     

      
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
      Smatrxx=1;                                //and nrj<>nr
      
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
      
   else      //entaglement
      Smatrxx=spzeros(matrxx);
      Smatrxy=spzeros(matrxx);
      Smatrxz=spzeros(matrxx);
      
      for j=2:natoms(2),    //start at atom #2 because the tip is atom #1
          Stmatrxx=sparse(S1(experiment.atom(1)));                                
      
          for i=2:(j-1),
              Stmatrxx=Stmatrxx.*.sparse(S1(experiment.atom(i)));
          end
          
          Stmatrxy=Stmatrxx;
          Stmatrxz=Stmatrxx;
      
          Stmatrxx=Stmatrxx.*.sparse(Sx(experiment.atom(j)));
          Stmatrxy=Stmatrxy.*.sparse(Sy(experiment.atom(j)));
          Stmatrxz=Stmatrxz.*.sparse(Sz(experiment.atom(j)));
          
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
 
  mone=ones(1,a(2));
  
  so1=list(matrxx,matrxy,matrxz,matrxu);
  so2=list(Smatrxx,Smatrxy,Smatrxz,Smatrxu);
  so3=so1;
  e=zeros(matrxx);er=e;en=e;enr=e;
  for j=1:4,
    for k=1:4,
      for l=1:4,

        e=e-e_sig(:,:,j,k,l).*(mone'.*.so3(l)(in,:)).*(so2(k)).'.*(mone.*.so1(j)(:,in));
        er=er-er_sig(:,:,j,k,l).*(mone'.*.so3(l)(in,:)).*(so1(k)).'.*(mone.*.so2(j)(:,in));
             
        en=en-en_sig(:,:,j,k,l).*(mone'.*.so1(l)(in,:)).*(so2(k)).'.*(mone.*.so3(j)(:,in));
        enr=enr-enr_sig(:,:,j,k,l).*(mone'.*.so1(l)(in,:)).*(so3(k)).'.*(mone.*.so2(j)(:,in));

      end
    end
  end
   
//  e=clean(e);
//  er=clean(er);
//  en=clean(en);
//  enr=clean(enr);
    
endfunction

function [e, er, en, enr]=e_epsilon_func_tip(experiment,in)
    // calculate the tunneling rate in 3rd order
    // via all possible tunneling pathes FOR THE ELECTRONS
    // output are 4 tensors (size 4x4x4) which give the probabiliy to tunnel from i->j->k
    // e is the 'normal' tensor for tunneling from tip to sample
    // er is the 'time-reversed' tensor for tunneling from tip to sample     
    // en is the 'normal' tensor for tunneling from sample to tip
    // enr is the 'time-reversed' tensor for tunneling from sample to tip
    // experiment has to contain:
    // 'experiment.ptip':     spin polarization of the tip
    // 'experiment.psample':  spin polarization of the sample 
   
    mone=ones(1,2);   
    nr_of_states=size(experiment.Eigenvec,2);
    e=zeros(nr_of_states,nr_of_states,4,4,4);er=e;en=e;enr=e;
    
    //e-transport tip->sample
    [x,y,z,u]=etransport_func_tip(experiment);
    
    //h-transport tip->sample
    tempexperiment.ptipK=-experiment.ptipK;
    tempexperiment.psampleK=-experiment.psampleK;
    tempexperiment.sef=experiment.sef;
    tempexperiment.psample=-experiment.psample
    tempexperiment.ptip=-experiment.ptip
    
    [xh,yh,zh,uh]=etransport_func_tip(tempexperiment);
       
    if experiment.jposition>1 then
        //3rd order scattering happens on sample
        
        //e-transport sample->sample
        tempexperiment.ptipK=experiment.psampleK;
        tempexperiment.psampleK=experiment.psampleK;
        tempexperiment.sef=experiment.sef;
        tempexperiment.psample=experiment.psample;
        tempexperiment.ptip=experiment.psample
        
        [xs,ys,zs,us]=etransport_func_tip(tempexperiment);
        
        //h-transport sample->sample
        tempexperiment.ptipK=-experiment.psampleK;
        tempexperiment.psampleK=-experiment.psampleK;
        tempexperiment.sef=experiment.sef;
        tempexperiment.psample=-experiment.psample;
        tempexperiment.ptip=-experiment.psample

        [xsh,ysh,zsh,ush]=etransport_func_tip(tempexperiment);
    else
        //3rd order scattering happens on tip        

        //e-transport tip->tip
        tempexperiment.ptipK=experiment.ptipK;
        tempexperiment.psampleK=experiment.ptipK;
        tempexperiment.sef=experiment.sef;
        tempexperiment.psample=experiment.ptip;
        tempexperiment.ptip=experiment.ptip;

        [xt,yt,zt,ut]=etransport_func_tip(tempexperiment);
        
        //h-transport tip->tip
        tempexperiment.ptipK=-experiment.ptipK;
        tempexperiment.psampleK=-experiment.ptipK;
        tempexperiment.sef=experiment.sef;
        tempexperiment.psample=-experiment.ptip;
        tempexperiment.ptip=-experiment.ptip;

        [xth,yth,zth,uth]=etransport_func_tip(tempexperiment);
    end

    for fin=1:nr_of_states,
       for mid=1:nr_of_states,

           if experiment.jposition>1 then
              so1=list(x(:,:,in,mid), y(:,:,in,mid), z(:,:,in,mid), u(:,:,in,mid));
              so2=list(xs(:,:,mid,fin), ys(:,:,mid,fin), zs(:,:,mid,fin), us(:,:,mid,fin));
              so3=list(x(:,:,in,fin)', y(:,:,in,fin)', z(:,:,in,fin)', u(:,:,in,fin)');
          
              so4=list(xh(:,:,in,mid), yh(:,:,in,mid), zh(:,:,in,mid), -uh(:,:,in,mid));  
              so5=list(xsh(:,:,mid,fin), ysh(:,:,mid,fin), zsh(:,:,mid,fin), -ush(:,:,mid,fin));  //
              so6=list(xh(:,:,in,fin)', yh(:,:,in,fin)', zh(:,:,in,fin)', -uh(:,:,in,fin)');

  
              // the following processes are calculated here with
              // excitations on a spin on the sample
              //
              // so1: an e scatters from t->s exciting a spin on the sample
              // so2: an e scatters from s->s exciting a spin on the sample
              // so3: an e scatters from s->t exciting a spin on the sample 
              //   transport reversal is accounted by the complex conjugated (')

              // so4: a h scatters from t->s exciting a spin on the sample
              // so5: a h scatters from s->s exciting a spin on the sample
              // so6: a h scatters from s->s exciting a spin on the sample
              //   transport reversal is accounted by the complex conjugated (')
              //
              // the inversion of U in so4-so6 is neccessary to account for hole transport
              
          else
              so1=list(xh(:,:,in,mid)', yh(:,:,in,mid)', zh(:,:,in,mid)', uh(:,:,in,mid)');
              so2=list(xth(:,:,mid,fin), yth(:,:,mid,fin), zth(:,:,mid,fin), uth(:,:,mid,fin));
              so3=list(xh(:,:,in,fin), yh(:,:,in,fin), zh(:,:,in,fin), uh(:,:,in,fin));

              so4=list(x(:,:,in,mid)', y(:,:,in,mid)', z(:,:,in,mid)', -u(:,:,in,mid)');  
              so5=list(xt(:,:,mid,fin), yt(:,:,mid,fin), zt(:,:,mid,fin), -ut(:,:,mid,fin));  //
              so6=list(x(:,:,in,fin), y(:,:,in,fin), z(:,:,in,fin), -u(:,:,in,fin));
              
              // the following processes are calculated here with
              // excitations on a spin on the sample
              //
              // so1: a h scatters from s->t exciting the spin on the tip
              //   transport reversal is accounted by the complex conjugated (')
              // so2: a h scatters from t->t exciting the spin on the tip
              // so3: a h scatters from t->s exciting the spin on the tip

              // so4: an e scatters from s->t exciting the spin on the tip
              //   transport reversal is accounted by the complex conjugated (')               
              // so5: an e scatters from t->t exciting a spin on the tip
              // so6: an e scatters from t->s exciting a spin on the tip
              //
              // the inversion of U in so4-so6 is neccessary to account for hole transport
          end
          
          for j=1:4,
            for k=1:4,
              for l=1:4,
                for i=1:2,          
                  temp=(mone'.*.so3(l)(i,:)).*(so2(k)).'.*(mone.*.so1(j)(:,i));  //should be +
                  e(mid,fin,j,k,l)=e(mid,fin,j,k,l)+sum(temp);
                        
                  temp=-(mone'.*.so3(l)(i,:)).*(mone.*.so1(k)(:,i)).*(so2(j).');  //should be -
                  er(mid,fin,j,k,l)=er(mid,fin,j,k,l)+sum(temp);
                        
                  temp=(mone'.*.so6(l)(i,:)).*(so5(k)).'.*(mone.*.so4(j)(:,i));  //should be +
                  en(mid,fin,j,k,l)=en(mid,fin,j,k,l)+sum(temp);
                        
                  temp=-(mone'.*.so6(l)(i,:)).*(mone.*.so4(k)(:,i)).*(so5(j).');   //should be -
                  enr(mid,fin,j,k,l)=enr(mid,fin,j,k,l)+sum(temp);
                end;
              end;
            end;
          end;
       end;
    end;
endfunction

function out=correlate(dm,t)
    //this fuction calculates the total correlation between two spins
    //
    //dm must be the desitymatrix of the bipartite system t
    dm1=traceout(dm,t,1);
    dm2=traceout(dm,t,2);
    
    out(1)=trace(dm*(Sx(t(1)).*.Sx(t(2))));
    out(2)=trace(dm*(Sy(t(1)).*.Sy(t(2))));
    out(3)=trace(dm*(Sz(t(1)).*.Sz(t(2))));
    
//    out(1)=trace(dm1*Sx(t(1)))*trace(dm2*Sx(t(2)));
//    out(2)=trace(dm1*Sy(t(1)))*trace(dm2*Sy(t(2)));
//    out(3)=trace(dm1*Sz(t(1)))*trace(dm2*Sz(t(2)));
  
    
    
endfunction

function out=correlate2(dm,t)
    //this fuction calculates the total correlation between to spins
    //reduced by the classical correlation.
    //
    //dm must be the desitymatrix of the bipartite system t
    dm1=traceout(dm,t,1);
    dm2=traceout(dm,t,2);
    
    out(1)=trace(dm1*Sx(t(1)))*trace(dm2*Sx(t(2)))-trace(dm*(Sx(t(1)).*.Sx(t(2))));
    out(2)=trace(dm1*Sy(t(1)))*trace(dm2*Sy(t(2)))-trace(dm*(Sy(t(1)).*.Sy(t(2))));
    out(3)=trace(dm1*Sz(t(1)))*trace(dm2*Sz(t(2)))-trace(dm*(Sz(t(1)).*.Sz(t(2))));
  
endfunction
