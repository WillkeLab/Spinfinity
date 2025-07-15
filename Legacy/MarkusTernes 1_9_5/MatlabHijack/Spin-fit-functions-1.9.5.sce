////////////////////////////////////////////////////////////////////////////////
//
//    Spin-fit-fuctions-1.9.5.sce
//
//    Stuttgart, Aachen, Jülich 07 Oktober 2021
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

function gui_fit()
    //routine called by the GUI
    //here we just check if we have data and decide if we fit spectroscopic or
    //state positional experimental data
    
    global %Data %Data_trunc experiment experiment2 calc guicheckbox1 guientry1 guislider1 gui old_k;

    //check if we have data
    if %Data==[] then 
        messagebox('No data to fit!', 'Attention!', "warning");
        abort;
    end;
    
    if size(%Data,2)==2 then
        fit_spec(); //we fit spectroscopic data, i.e. I(V), dI/dV(V), or d2I/dV2(V) data
    else
        fit_states(); //we fit state position data, i.e. e_n(B) data
    end
endfunction

function fit_states()
        
    global %Data experiment experiment2 calc guicheckbox1 guientry1 guislider1 minvalues1 maxvalues1 gui;
    
    if size(%Data,2)>(2*experiment.atom.S)+1 then
        messagebox('Spin must be higher!', 'Attention!', "warning");
        abort;
    end
    
    if size(experiment.atom,2)>1 then
        messagebox('This function is not yet supported', 'Attention!', "warning");
        abort;
    end
     
    %Data(:,2:$)=gsort(%Data(:,2:$),'c','i');
        // menue
    sig = x_mdialog('Fit parameter',..
    ['maximum number of calls';'maximum number of iterations'],['100';'50']);
    if prod(size(sig)<>[2,1]) then
        abort;
    end;
    
    atom=experiment.atom(experiment.position);
    i=1;
    fitmin=[];
    fitmax=[];
    if experiment.sw(1) then
       k(i)=atom.g;
       fitmin(i)=minvalues1(1);
       fitmax(i)=maxvalues1(1);
       i=i+1;
    end
    if atom.use_Stevens == 0 then
       if experiment.sw(2) then
           k(i)=atom.D;
           fitmin(i)=minvalues1(2);
           fitmax(i)=maxvalues1(2);
           i=i+1;
       end
       if experiment.sw(3) then
           k(i)=atom.E;
           fitmin(i)=minvalues1(3);
           fitmax(i)=maxvalues1(3);
           i=i+1;
       end
       if experiment.sw(4) then
           k(i)=atom.alpha;
           fitmin(i)=minvalues1(4);
           fitmax(i)=maxvalues1(4);
           i=i+1;
       end
       if experiment.sw(5) then
           k(i)=atom.beta;
           fitmin(i)=minvalues1(5);
           fitmax(i)=maxvalues1(5);
           i=i+1;
       end
       if experiment.sw(6) then
           k(i)=atom.gamma;
           fitmin(i)=minvalues1(6);
           fitmax(i)=maxvalues1(6);
           i=i+1;
       end

    else
       for j=1:size(%Stevstr,2),
           if experiment.sw(1+j) then
               a=evstr(part(%Stevstr(j),4))/2;
               b=evstr(part(%Stevstr(j),6))+1;
               k(i)=atom.Stevens(a,b);
               fitmin(i)=minvalues1(1+j);
               fitmax(i)=maxvalues1(1+j);
               i=i+1;
           end
       end
       if experiment.sw(j+2) then
           k(i)=atom.alpha;
           fitmin(i)=minvalues1(j+2);
           fitmax(i)=maxvalues1(j+2);
           i=i+1;
       end
       if experiment.sw(j+3) then
           k(i)=atom.beta;
           fitmin(i)=minvalues1(j+3);
           fitmax(i)=maxvalues1(j+3);
           i=i+1;
       end
       if experiment.sw(j+4) then
           k(i)=atom.gamma;
           fitmin(i)=minvalues1(j+4);
           fitmax(i)=maxvalues1(j+4);
           i=i+1;
       end
    end;
    
    k0=k;
    winId=progressionbar('Fitting. Please be patient!');
    
    vers_nr = getversion('scilab');
    if vers_nr(1)<6 then
        [fopt,k,gopt]=leastsq(imp=2,diff_states,"b",fitmin,fitmax,k0,"ar",evstr(sig(1)),evstr(sig(2)));
        else
        [fopt,k,gopt]=leastsq(2,diff_states,"b",fitmin,fitmax,k0,"ar",evstr(sig(1)),evstr(sig(2)));
        // version 6 has a different iprint syntax
    end
    close(winId);
    
    atom=experiment.atom(experiment.position);
    i=1;
    if experiment.sw(1) then
       atom.g=k(i);i=i+1;
    end
    if atom.use_Stevens == 0 then
        if experiment.sw(2) then
            atom.D=k(i);i=i+1;
        end
        if experiment.sw(3) then
            atom.E=k(i);i=i+1;
        end
        if experiment.sw(4) then
            atom.alpha=k(i);i=i+1;
        end
        if experiment.sw(5) then
            atom.beta=k(i);i=i+1;
        end
        if experiment.sw(6) then
            atom.gamma=k(i);i=i+1;
        end

    else
        for j=1:size(%Stevstr,2),
            if experiment.sw(1+j) then
                a=evstr(part(%Stevstr(j),4))/2;
                b=evstr(part(%Stevstr(j),6))+1;
                atom.Stevens(a,b)=k(i);
                i=i+1;
            end
            if experiment.sw(j+2) then
                atom.alpha=k(i);i=i+1;
            end
            if experiment.sw(j+3) then
                atom.beta=k(i);i=i+1;
            end
            if experiment.sw(j+4) then
                atom.gamma=k(i);i=i+1;
            end

        end
    end;
   
    experiment.atom(experiment.position)=atom;
    experiment2=experiment;
    gui_switchexp();
    cleardata(2);   //clear all calculation
    gui_drawstates();
        
endfunction

function out=diff_states(k)
    
    global %Data experiment;
    
    atom=experiment.atom;
    i=1;
    if experiment.sw(1) then
       atom.g=k(i);i=i+1;
    end
    if atom.use_Stevens == 0 then
        if experiment.sw(2) then
            atom.D=k(i);i=i+1;
        end
        if experiment.sw(3) then
            atom.E=k(i);i=i+1;
        end
        if experiment.sw(4) then
            atom.alpha=k(i);i=i+1;
        end
        if experiment.sw(5) then
            atom.beta=k(i);i=i+1;
        end
        if experiment.sw(6) then
            atom.gamma=k(i);i=i+1;
        end

    else
        for j=1:size(%Stevstr,2),
            if experiment.sw(1+j) then
                a=evstr(part(%Stevstr(j),4))/2;
                b=evstr(part(%Stevstr(j),6))+1;
                atom.Stevens(a,b)=k(i);
                i=i+1;
            end
        end
        if experiment.sw(j+2) then
            atom.alpha=k(i);i=i+1;
        end
        if experiment.sw(j+3) then
            atom.beta=k(i);i=i+1;
        end
        if experiment.sw(j+4) then
            atom.gamma=k(i);i=i+1;
        end

    end;
    out=[];
    for Bz=1:size(%Data,1),
        B=[0,0,%Data(Bz,1)];
        
        [M,vm]=spec(HZ(atom,B)+Haniso(atom));
        eigenval=diag(vm);
        eigenval=eigenval-min(eigenval);
        for i=2:size(%Data,2),
            out=[out;%Data(Bz,i)-eigenval(i)];
        end
    end
    
endfunction

function [k,fitmin,fitmax]=exp_to_k(experiment)
    
    //Determine the initial values "k" and the limits "fitmin", "fitmax"
    //from the  values of "experiment.sw"
    //the order is as following if sw is set:
    
   // atom.g 
   // atom.D    (only if "expeiment.atom.use_Stevens"==0)
   // atom.E    (only if "expeiment.atom.use_Stevens"==0)
   // atom.alpha 
   // atom.beta       
   // atom.gamma      
   // experiment.heisenberg_coupling    (only if we have two atoms )
   // atom.U
   // atom.J
   // atom.w
   // experiment.x0
   // experiment.T
   // experiment.ptip (only if ptip<>0 and B<>0 OR when we use a func tip and
   //                 have correlations in tip and we are NOT on the tip site
   // experiment.A
   // experiment.b
   // experiment.y0
    
    global minvalues1 maxvalues1 %Stevstr gui;
    
    i=1;k=[];fitmin=[];fitmax=[];
//    atom=experiment.atom(experiment.position);
    atom=experiment.atom(gui.sel_spin.value);
    
    if experiment.sw(1) then
       k(i)=atom.g;i=i+1;
       fitmin=[fitmin;minvalues1(1)];
       fitmax=[fitmax;maxvalues1(1)];
    end

    if atom.use_Stevens == 0 then
        if experiment.sw(2) then
            k(i)=atom.D;
            fitmin(i)=minvalues1(2);
            fitmax(i)=maxvalues1(2);
            i=i+1;
        end
        if experiment.sw(3) then
            k(i)=atom.E;
            fitmin(i)=minvalues1(3);
            fitmax(i)=maxvalues1(3);
            i=i+1;
        end
       if experiment.sw(4) then
           k(i)=atom.alpha;i=i+1;
           fitmin=[fitmin;minvalues1(4)];
           fitmax=[fitmax;maxvalues1(4)];
       end
       if experiment.sw(5) then
           k(i)=atom.beta;i=i+1;
           fitmin=[fitmin;minvalues1(5)];
           fitmax=[fitmax;maxvalues1(5)];
       end
       if experiment.sw(6) then
           k(i)=atom.gamma;i=i+1;
           fitmin=[fitmin;minvalues1(6)];
           fitmax=[fitmax;maxvalues1(6)];
       end
       index=7;
    else
       for j=1:size(%Stevstr,2),
          if experiment.sw(1+j) then
              a=evstr(part(%Stevstr(j),4))/2;
              b=evstr(part(%Stevstr(j),6))+1;
              k(i)=atom.Stevens(a,b);
              fitmin(i)=minvalues1(1+j);
              fitmax(i)=maxvalues1(1+j);
              i=i+1;
          end
       end
       index=2+size(%Stevstr,2);
       if experiment.sw(index) then
           k(i)=atom.alpha;i=i+1;
           fitmin=[fitmin;minvalues1(index)];
           fitmax=[fitmax;maxvalues1(index)];
       end
       if experiment.sw(index+1) then
           k(i)=atom.beta;i=i+1;
           fitmin=[fitmin;minvalues1(index+1)];
           fitmax=[fitmax;maxvalues1(index+1)];
       end
       if experiment.sw(index+2) then
           k(i)=atom.gamma;i=i+1;
           fitmin=[fitmin;minvalues1(index+2)];
           fitmax=[fitmax;maxvalues1(index+2)];
       end

       index=index+3;
    end;

    if size(experiment.atom,2) >1 & experiment.sw(index+9) then
       k(i)=experiment.heisenberg_coupling;
       fitmin(i)=-%inf;
       fitmax(i)=+%inf;
       i=i+1;
    end
    
    if experiment.sw(index) then  
       k(i)=atom.U;
       fitmin(i)=minvalues1(index);
       fitmax(i)=maxvalues1(index);
       i=i+1;
    end

    if experiment.sw(index+1) then
       k(i)=atom.J;
       fitmin(i)=minvalues1(index+1);
       fitmax(i)=maxvalues1(index+1);
       i=i+1;
    end

    if experiment.sw(index+2) then
       k(i)=atom.w;
       fitmin(i)=minvalues1(index+2);
       fitmax(i)=maxvalues1(index+2);
       i=i+1;
    end
    
    if experiment.sw(index+5) then
       k(i)=experiment.x0;
       fitmin(i)=minvalues1(index+5);
       fitmax(i)=maxvalues1(index+5);
       i=i+1;
    end
    
    if experiment.sw(index+7) then
       k(i)=experiment.T;
       fitmin(i)=minvalues1(index+7)*%kBoltzmann;   //go from K to meV
       fitmax(i)=maxvalues1(index+7)*%kBoltzmann;
       i=i+1;
    end
    
    if experiment.sw(index+8) & experiment.func_tip>0 &..
       gui.sel_spin.value > 1 & experiment.corr_tip then
       //in the case that the spin-pol switch is set AND 
       //we have a functionalized tip AND we are not on the first (tip) spin AND
       //tip correlation is ON than the Spin POL controls the coupling of the TIP
       //to the reservoir, i.e. experiment.atom.J(1)
       k(i)=experiment.atom(1).J;
       fitmin(i)=minvalues1(index+8);
       fitmax(i)=maxvalues1(index+8);
       i=i+1;
    elseif experiment.sw(index+8) & sum(experiment.B.^2)<>0 & sum(experiment.ptip.^2)<>0 then
       //calculate angle between B and ptip
       angle=real(acos((experiment.ptip*experiment.B')/...
       (sqrt(sum(experiment.ptip.^2))*sqrt(sum(experiment.B.^2)))));
       //if angle is larger than 90 deg eta<0
       //eta is the length of experiment.ptip
       if angle>%pi/2 then
          k(i)=-sqrt(sum(experiment.ptip.^2));
       else
          k(i)=sqrt(sum(experiment.ptip.^2));
       end
       fitmin(i)=minvalues1(index+8);
       fitmax(i)=maxvalues1(index+8);
       i=i+1;
    end
    
    if experiment.sw(index+3) then
       k(i)=experiment.A;
       fitmin(i)=minvalues1(index+3);
       fitmax(i)=maxvalues1(index+3);
       i=i+1;
    end
    
    if experiment.sw(index+4) then
       k(i)=experiment.b;
       fitmin(i)=minvalues1(index+4);
       fitmax(i)=maxvalues1(index+4);
       i=i+1;
    end
    
    if experiment.sw(index+6) then
       k(i)=experiment.y0;
       fitmin(i)=minvalues1(index+6);
       fitmax(i)=maxvalues1(index+6);
       i=i+1;
    end
    
endfunction

function [experiment]=k_to_exp(k,experiment)

   //this function plays back the fitted values contained in the vector "k" 
   //into the correctly determied positions of the "experiment". For this the 
   //""experiment.sw" data is used to determine which values have been fit.
   //the order is the following:
       
   // atom.g 
   // atom.D    (only if "expeiment.atom.use_Stevens"==0)
   // atom.E    (only if "expeiment.atom.use_Stevens"==0)
   // atom.alpha 
   // atom.beta       
   // atom.gamma      
   // experiment.heisenberg_coupling    (only if we have two atoms )
   // atom.U
   // atom.J
   // atom.w
   // experiment.x0
   // experiment.T
   // experiment.ptip (only if ptip<>0 and B<>0 OR when we use a func tip and
   //                 have correlations in tip and we are NOT on the tip site
   // experiment.A
   // experiment.b
   // experiment.y0
    
    global %Stevstr gui;
    
    atom=experiment.atom(gui.sel_spin.value);
    i=1;
    if experiment.sw(1) then
       atom.g=k(i);i=i+1;
    end
    
    if atom.use_Stevens == 0 then
        if experiment.sw(2) then
            atom.D=k(i);i=i+1;
        end
        if experiment.sw(3) then
            atom.E=k(i);i=i+1;
        end
        if experiment.sw(4) then
            atom.alpha=k(i);i=i+1;
        end
        if experiment.sw(5) then
            atom.beta=k(i);i=i+1;
        end
        if experiment.sw(6) then
            atom.gamma=k(i);i=i+1;
        end
        
        index=7;
    else
        for j=1:size(%Stevstr,2),
            if experiment.sw(1+j) then
                a=evstr(part(%Stevstr(j),4))/2;
                b=evstr(part(%Stevstr(j),6))+1;
                atom.Stevens(a,b)=k(i);
                i=i+1;
            end
        end
        
        index=2+size(%Stevstr,2);
        
        if experiment.sw(index) then
            atom.alpha=k(i);i=i+1;
        end
        if experiment.sw(index+1) then
            atom.beta=k(i);i=i+1;
        end
        if experiment.sw(index+2) then
            atom.gamma=k(i);i=i+1;
        end

        index=index+3;
    end;
   
    if size(experiment.atom,2) >1 & experiment.sw(index+9) then
       experiment.heisenberg_coupling=k(i);
       i=i+1;
    end
    if experiment.sw(index) then
       atom.U=k(i);i=i+1;
    end
    if experiment.sw(index+1) then
       atom.J=k(i);i=i+1;
    end
    if experiment.sw(index+2) then
       atom.w=k(i);i=i+1;
    end
    if experiment.sw(index+5) then
       experiment.x0=k(i);i=i+1;
    end
    if experiment.sw(index+7) then
       experiment.T=k(i);i=i+1;
    end
    if experiment.sw(index+8) & experiment.func_tip>0 &..
       gui.sel_spin.value > 1 & experiment.corr_tip then
       //in the case that the spin-pol switch is set AND 
       //we have a functionalized tip AND we are not on the first (tip) spin AND
       //tip correlation is ON than the Spin POL controls the coupling of the TIP
       //to the reservoir, i.e. experiment.atom.J(1)
       experiment.atom(1).J=k(i);
       i=i+1;
    elseif experiment.sw(index+8) & sum(experiment.B.^2) <>0 & sum(experiment.ptip.^2)<>0 then
       angle=real(acos((experiment.ptip*experiment.B')/...
       (sqrt(sum(experiment.ptip.^2))*sqrt(sum(experiment.B.^2)))));
       sig=1-2*(angle>%pi/2); //if angle>pi/2 then sig=-1, else +1
       experiment.ptip=experiment.ptip/sqrt(sum(experiment.ptip.^2))*k(i)*sig;
       i=i+1;
    end
    if experiment.sw(index+3) then
       experiment.A=k(i);i=i+1;
    end
    if experiment.sw(index+4) then
       experiment.b=k(i);i=i+1;
    end
    if experiment.sw(index+6) then
       experiment.y0=k(i);i=i+1;
    end
        
    experiment.atom(gui.sel_spin.value)=atom;
    
endfunction

function k=fit_spec(varargin)
    
   //the main fit function forspectroscopic data
   //the global variable "experiment" is neccessary
   //"varargin" can contain the 'maximum number of calls' and
   //the'maximum number of iterations'. If not given a popup will ask the values
   // the output is k, the fitted values

    global %Data %Data_trunc experiment experiment2 calc old_k itcount gui minvalues1 maxvalues1;
    
    experiment2=experiment;
    //truncate data if neccessary
    if max(abs(%Data(:,1)))>abs(experiment.xrange) then
        %Data_trunc=%Data(find(abs(%Data(:,1))<=abs(experiment.xrange)),:)
    else
        %Data_trunc=%Data;
    end
    
    [lhs,rhs]=argn(0); 
    
    if rhs < 2 then  
        // ask for number of iteration menue
        sig = x_mdialog('Fit parameter',..
        ['maximum number of calls';'maximum number of iterations'],['100';'50']);
        if prod(size(sig)<>[2,1]) then
            abort;
        end;
        sig(~isnum(sig))='10'; //replace bullshit
        sig=[evstr(sig(1)),evstr(sig(2))];
    else
        sig=[varargin(1),varargin(2)];
    end
    experiment.position=gui.sel_spin.value;
    experiment.jposition=experiment.position;
    
    experiment.allatomsequ=gui.allatomsequ.Value;
    
    [k,fitmin,fitmax]=exp_to_k(experiment);
    
    k0=k;old_k=[];calc.spec_raw=[];
    
    fst=create_f_fit(experiment);
    //disp(k0);
    execstr(fst);
    
    function [fk,dfk,ind]=costfkt(k,ind)
        global %Data_trunc experiment old_k calc itcount;
        if %stopcalc then
            fk=0;
            dfk=zeros(k);
            return;
        end  
                  
        fk=f_fit(k);itcount(3)=fk;
        dfk=numderivative(f_fit,k);           
        if exists('itcount') then
            itcount(1)=itcount(1)+1;
        else
            itcount=[1,1,0];
        end;
    endfunction

    
    itcount=[0,0,0];
    gui_disable();
    tic();
    winId=progressionbar('Fitting. Please be patient!');
    gui_startcalc();

//   [fopt, k, gopt] = optim(costfkt, k0,"ar",sig(1),sig(2),imp=2);
    disp([fitmin,k,fitmax]);
    
    vers_nr = getversion('scilab');
    if vers_nr(1)<6 then
          [fopt, k, gopt] = optim(costfkt, "b", fitmin, fitmax, k0,"ar",sig(1),sig(2),imp=2);
       else
          [fopt, k, gopt] = optim(costfkt, "b", fitmin, fitmax, k0,"ar",sig(1),sig(2),iprint=2);      
    end

    close(winId);
    
    //end of the fit
   
    //estimating the error by calculating the Hessian matrix:
    [dfk,hes]=numderivative(f_fit,k);
    //select the diagonal elements
    hes=matrix(hes,sqrt(length(hes)),sqrt(length(hes)));
    hes=diag(hes);

    //calculate the error of the fit:
    data_length=size(%Data_trunc,1);
    number_fitvar=length(k);
    
    ferror=sqrt((fopt)/(data_length-number_fitvar)*(hes.^-1));
        
    if rhs <> 2 then
        messagebox(['Fitting finshed!';..
           'Elapsed time: '+string(toc())+' s';..
           'No of calls: '+string(itcount(1));..
           'chi^2 Error: '+string(fopt);..
           'Parameters:';..
           string(k)+" +/- "+string(ferror)],..
           'Results of the fit','info');
        if or(k<=fitmin+1E-10) | or(k>=fitmax-1E-10) then
             messagebox(['One or more parameters are close to their limits of the fitrange!',..
             'Consider to change the limits and repeat fit.']..
             ,'Attention','warning');
        end   
    end;
    gui_stopcalc();

    disp("old -> new parameter values | errors");
    disp([k0,k,ferror]);
    
    experiment=k_to_exp(k,experiment);
    
    experiment.xrange=experiment2.xrange;
   
    update_gui_values();
    gui_enable();
    
   // cleardata(2);   //clear all calculation
    gui_drawspec();
    
    //if ~experiment.rate_calc then
        gui.graph_ui.children(2).children.foreground=5;
        gui.graph_ui.children(2).children.thickness=2;
    //end
    //sound(loadwave("SCI/modules/sound/tests/nonreg_tests/bug_629.wav"));
endfunction

function [fst]=create_f_fit(experiment) 
    //here we create dynamically the fit function [fst]
    //[fst] is a string which can be executed as script 
    
    global %renormalize_E %Stevstr;
    
    fst=['function [fk]=f_fit(k),';..
    ' ';..
    'global %renormalize_E %Data_trunc experiment old_k calc fitres;';..
    'atom=experiment.atom('+string(experiment.position)+');'];
    
    fst1="";    //in this string we put all comparisons of the form "k(x)<>k_old(x)" 
                //which when changed need a recalculation of the Hamilton and the spectrum
                //this is the case if "g","D","E", "alpha", "beta", "gamma", or 
                //one of the Stevens parameter change.
                
    fst2="";    //in this string we put all comparisons of the form "k(x)<>k_old(x)" 
                //which when changed need a recalculation of the spectrum
                //this is the case if "spin-pol", "T", "w", "U", "J", "x0", or "A" if rate_equ=%T change.
                
    fst3="";    //in this string we put all comparisons of the form "k(x)<>k_old(x)" 
                //which when changed only need a rescale
                //this is the case if "b", "y0", or "A" if rate_equ=%F change.
    
    //dividing the comparison into these 3 cases dramatically speed up calculation.
    
    i=1;    //counter which takes care that we have the order of the parameters k(i) correct
    if experiment.sw(1) then
       fst=[fst;'atom.g=k('+string(i)+');'];
       fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
       i=i+1;
    end
    if experiment.atom(experiment.position).use_Stevens == 0 then
       if experiment.sw(2) then
           fst=[fst;'atom.D=k('+string(i)+');'];
           fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
           i=i+1;
        end
        if experiment.sw(3) then
           fst=[fst;'atom.E=k('+string(i)+');'];
           fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
           i=i+1;
        end
        if experiment.sw(4) then
           fst=[fst;'atom.alpha=k('+string(i)+');'];
           fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
           i=i+1;
        end
        if experiment.sw(5) then
           fst=[fst;'atom.beta=k('+string(i)+');'];
           fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
           i=i+1;
        end
        if experiment.sw(6) then
           fst=[fst;'atom.gamma=k('+string(i)+');'];
           fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
           i=i+1;
        end
        index=7;    //the "U" parameter is at position 7
    else
        for j=1:size(%Stevstr,2),
            if experiment.sw(1+j) then
                a=evstr(part(%Stevstr(j),4))/2;
                b=evstr(part(%Stevstr(j),6))+1;
                fst=[fst;'atom.Stevens('+string(a)+','+string(b)+')=k('+string(i)+');'];
                fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
                i=i+1;
            end
        end
        index=2+size(%Stevstr,2); //set index to the "theta" parameter
        if experiment.sw(index) then
           fst=[fst;'atom.alpha=k('+string(i)+');'];
           fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
           i=i+1;
        end
        if experiment.sw(index+1) then
           fst=[fst;'atom.beta=k('+string(i)+');'];
           fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
           i=i+1;
        end
        if experiment.sw(index+2) then
           fst=[fst;'atom.gamma=k('+string(i)+');'];
           fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
           i=i+1;
        end
        
        index=index+3; //set index to the "U" parameter
    end;

    if experiment.sw(index+9) & size(experiment.atom,2) >1 then
       fst=[fst;'experiment.heisenberg_coupling=k('+string(i)+');'];
       fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
       i=i+1;
    end
    
    if experiment.sw(index) then
       fst=[fst;'atom.U=k('+string(i)+');'];
       fst2=[fst2;'k('+string(i)+')<>old_k('+string(i)+')'];
       i=i+1;
    end
    if experiment.sw(index+1) then
       fst=[fst;'atom.J=k('+string(i)+');'];
       if exists("%renormalize_E") then
           if %renormalize_E then
               fst1=[fst1;'k('+string(i)+')<>old_k('+string(i)+')'];
           else
               fst2=[fst2;'k('+string(i)+')<>old_k('+string(i)+')'];
           end,
       else
           fst2=[fst2;'k('+string(i)+')<>old_k('+string(i)+')'];
       end;
       i=i+1;
    end
    if experiment.sw(index+2) then
       fst=[fst;'atom.w=k('+string(i)+');'];
       fst2=[fst2;'k('+string(i)+')<>old_k('+string(i)+')'];
       i=i+1;
    end
    if experiment.sw(index+5) then
       fst=[fst;'experiment.x0=k('+string(i)+');'];
       fst2=[fst2;'k('+string(i)+')<>old_k('+string(i)+')'];
       i=i+1;
    end
    if experiment.sw(index+7) then
       fst=[fst;'experiment.T=k('+string(i)+');'];
       fst2=[fst2;'k('+string(i)+')<>old_k('+string(i)+')'];
       i=i+1;
    end
    if experiment.sw(index+8) & experiment.func_tip>0 &..
       experiment.position > 1 & experiment.corr_tip then
       //in the case that the spin-pol switch is set AND 
       //we have a functionalized tip AND we are not on the first (tip) spin AND
       //tip correlation is ON than the Spin POL controls the coupling of the TIP
       //to the reservoir, i.e. experiment.atom.J(1)
       fst=[fst;'experiment.atom(1).J=k('+string(i)+');'];
       fst2=[fst2;'k('+string(i)+')<>old_k('+string(i)+')'];
       i=i+1;
    elseif experiment.sw(index+8) & sum(experiment.B.^2) <> 0 & sum(experiment.ptip.^2)<>0 then
       //expected bahavior which allow to fit the tip's spin-poarisation
       fst=[fst;'angle=real(acos((experiment.ptip*experiment.B'')/(sqrt(sum(experiment.ptip.^2))*sqrt(sum(experiment.B.^2)))));'];
       fst=[fst;'sig=1-2*(angle>%pi/2);'];
       fst=[fst;'experiment.ptip=experiment.ptip/sqrt(sum(experiment.ptip.^2))*k('+string(i)+')*sig;'];
       fst2=[fst2;'k('+string(i)+')<>old_k('+string(i)+')'];
       i=i+1;
    end

    if experiment.sw(index+3) then
       fst=[fst;'experiment.A=k('+string(i)+');'];
       if experiment.rate_calc then
           fst2=[fst2;'k('+string(i)+')<>old_k('+string(i)+')'];
       end;
       i=i+1;
    end
    if experiment.sw(index+4) then
       fst=[fst;'experiment.b=k('+string(i)+');'];
       i=i+1;
    end
    if experiment.sw(index+6) then
       fst=[fst;'experiment.y0=k('+string(i)+');'];
       i=i+1;
    end
    
    if or(fst1<>"") then
        fst1=fst1(2:$); //delete first line which is blank
        fst1=fst1+'|';  // add the logical "or" to each lime 
        fst1=strcat(fst1); //concatenate fst1 to something like "k(1)<>k_old(1)|k(2)<>k_old(2)"
    end
    
    if or(fst2<>"") then
        fst2=fst2(2:$); //delete first line
        fst2=fst2+'|';
        fst2=strcat(fst2);
    end
    
    if experiment.allatomsequ==%F | experiment.allatomsequ == 0 then
        fst=[fst;..
        'experiment.atom('+string(experiment.position)+')=atom;';..
        'c=%F;'];
    else
        fst=[fst;..
        'experiment.atom(:)=atom;';..
        'c=%F;'];
    end

    fst=[fst;.. //check if we have to recalculate all
    'if '+fst1+' experiment.Eigenval==[] then';..
    '   [M,vm]=spec(full(hamiltonian(experiment)));';..
    '   experiment.Eigenvec=M;';..
    '   experiment.Eigenval=diag(vm);';..
    '   c=%T;';..
    'end']
    
    if experiment.rate_calc then
        fst=[fst;..
        '//rate calculation';..
        'if '+fst2+' calc.spec_raw==[] | c then';..
        '   vm=experiment.Eigenval';..
        '   ediff=ones(1:size(vm,1))''.*.vm''-ones(1:size(vm,1)).*.vm;'];
        
        if experiment.rate_calc==2 then
            fst=[fst;..
            '   J2_t=experiment.A;J_t=-sqrt(J2_t);']; //normal tip
        end
        
        if ~experiment.func_tip then
            fst=[fst;..
            '   rate_ts=Rate2nd2(experiment);']; //normal tip
        elseif experiment.func_tip==1 then 
            fst=[fst;.. //functionalzed dark tip (version 1.7.1) 
            '   //dark functionalized tip';..
            '   posbkup=experiment.position';..        //save the parameter page
            '   experiment.position=2;';..             //but always calculate the visible spin
            '   rate_ts=Rate2nd2_func_tip(experiment);'];
        else
            fst=[fst;.. //functionalzed bright tip (version 1.7.1) 
            '   //bright tip';..
            '   posbkup=experiment.position';..        //save the parameter page
            '   experiment.position=2;';..             //but always calculate the visible spin
            '   rate_ts=Rate2nd2_func_tip_2(experiment);'];
        end

        fst=[fst;..
        '   rate_st=rate_ts'' ';..
        '   tempexperiment=experiment;';..
        '   tempexperiment.ptip=experiment.psample;';..
        '   tempexperiment.ptipK=tempexperiment.psampleK;';..
        '   tempexperiment.atom(:).U=0;'];
        
        if experiment.use_T_ts then
            fst=[fst;..
            '   rate_ss=Rate2nd2(tempexperiment);']; //the scaling is done in Rate2nd2 (version 1.6))        
        else
            fst=[fst;..
            '   rate_ss=tempexperiment.atom(experiment.position).J^2*Rate2nd2(tempexperiment);']; 
        end
        
        if experiment.rate_calc==2 then
            fst=[fst;..
            '   tempexperiment=experiment;';..
            '   tempexperiment.psample=experiment.ptip;';..
            '   tempexperiment.psampleK=tempexperiment.ptipK;';..
            '   tempexperiment.atom(:).U=0;';..
            '   rate_tt=J2_t*Rate2nd2(tempexperiment);';..
            '   rate2_tt=(fbox((ediff)/experiment.T)*experiment.T).*rate_tt;'];
        end
        
        if experiment.entanglement & size(experiment.atom,2) > 1 then
            atms=[1:nbr_of_atoms];
            atms(experiment.position)=[]; //we want to sum over all atoms except the one we are measuring
            if find(atms==1) & experiment.func_tip>0 then
                functip=%T; //set marker for functionalized tip
                atms=atms(2:$); //remove the first element
            else
                functip=%F;
            end
            
            
            fst=[fst;..
            '   for atnr=['+strcat(string(atms),',')+']';..
            '      tempexperiment.position=atnr;'];
            if experiment.use_T_ts then
                fst=[fst;..
                '      rate_ss=rate_ss+Rate2nd2(tempexperiment);';..
                '   end']; //new (version 1.6)
            else
                fst=[fst;..
                '      rate_ss=rate_ss+tempexperiment.atom(atnr).J^2*Rate2nd2(tempexperiment);';..
                '   end']; 
            end
        end

                //include the first spin as tip spin
//              '   tempexperiment.position=1;'];
//               '   tempexperiment.ptip=experiment.psample;';..               



        fst=[fst;..
        '   rate2_ss=(fbox((ediff)/experiment.T)*experiment.T).*rate_ss;';..
        '   i=0;';..
        '   x=%Data_trunc(:,1)-experiment.x0;';..   //shift the x-values by x0
        '   dx=(x($)-x(1))/(length(x)-1);';];       //determine the differerence between datapoints
        if experiment.rate_calc==1 then
            fst=[fst;..
            '   J2_ts=experiment.A;';];  
        elseif experiment.rate_calc==2 then 
            fst=[fst;..
            '   J2_ts=experiment.atom(experiment.position).J^2*experiment.A/(experiment.atom(experiment.position).J^2+experiment.A)';];  
        end

           
        if experiment.spectrum_deriv==1 then
            fst=[fst;..
            '      xx=[x;x($)+dx]-dx/2;'] //expand the dataset by one point
        elseif experiment.spectrum_deriv==2 then
            fst=[fst;..
            '   //2nd derivative';..
            '      xx=[x;x($)+dx;x($)+2*dx]-dx;'];     //expand the dataset by two points
        else
            fst=[fst;..
            '      xx=x;'];
        end

 // Include 3rd order
 
        if experiment.third_order_calc then
            fst=[fst;..
            '   //3rd order calc';..
            '   experiment.jposition=experiment.position;';..
            '   bk_xr=experiment.xrange;';..
            '   bk_sl=experiment.spec_length;';..
            '   experiment.xrange=xx;';..
            '   experiment.spec_length=length(xx);';..
            ''];
            if experiment.use_T_ts then
                fst=[fst;..
                '   [rate3_st,rate3_ts]=spec3rate(experiment,%F);';..
                ];
            else
                fst=[fst;..
                '   [rate3_st,rate3_ts]=spec3rate(experiment, %F);';..
                '   rate3_st=rate3_st*J2_ts;';..
                '   rate3_ts=rate3_ts*J2_ts;';..
                ];
            end
            if experiment.rate_calc==2 then
                fst=[fst;..
                '   tempexperiment=experiment;';..
                '   tempexperiment.ptip=experiment.psample;';..
                '   tempexperiment.psample=experiment.ptip;';..
                '   tempexperiment.entanglement=%f;';..
                '   tempexperiment.atom(experiment.position).J=J_t;';..
                '   tempexperiment.xrange=-experiment.xrange;';..
                '   [rate3t_ts,rate3t_st]=spec3rate(tempexperiment,%F);';..//invert direction
                '   rate3t_st=rate3t_st*J2_ts;';..
                '   rate3t_ts=rate3t_ts*J2_ts;';..
                '   rate3_tt=spec3rate_tt(experiment);';..
                ];
            end
            
            fst=[fst;..
            '   rate3_ss=spec3rate_ss(experiment);';..
            '   experiment.xrange=bk_xr;';..
            '   experiment.spec_length=bk_sl;';..
            ];
            
        end


        fst=[fst;..
        '   for U=xx'',';..
        '      i=i+1;';..
        '      rate2_ts=J2_ts*fbox((ediff+U)/experiment.T)*experiment.T.*rate_ts;';..
        '      rate2_st=J2_ts*fbox((ediff-U)/experiment.T)*experiment.T.*rate_st;';..
        ];
        if experiment.third_order_calc then
            if experiment.rate_calc==1 then
                fst=[fst;..
                '       rate2=rate2_ss+rate3_ss+rate2_ts+rate3_ts(:,:,i)+rate2_st+rate3_st(:,:,i);';..
                ];
            elseif experiment.rate_calc==2 then
                fst=[fst;..
                '       rate2=rate2_ss+rate2_tt+rate3_ss+rate3_tt+rate2_ts+rate3_ts(:,:,i)+rate3t_ts(:,:,i)+rate2_st+rate3_st(:,:,i)+rate3t_st(:,:,i);';..
                ];
            end
            
        else
            if experiment.rate_calc==1 then
                fst=[fst;..
                '       rate2=rate2_ss+rate2_ts+rate2_st;';..
                ];
            else
                fst=[fst;..
                '       rate2=rate2_ss+rate2_tt+rate2_ts+rate2_st;';..
                ];
            end
            
        end
        fst=[fst;..
        '      rate21=rate2-eye(rate2).*rate2-diag(sum(rate2-eye(rate2).*rate2,1));';..
        '      [ststsol]=kernel(rate21);';..
        '      sc=sum(ststsol,1);';..
        '      stst=ststsol(:,1)/sc(1);';..
        ]
        if experiment.third_order_calc then
            if experiment.rate_calc==1 then
                fst=[fst;..
                '      I(i)=sum(rate2_ts*stst)+1*sum(rate3_ts(:,:,i)*stst)-sum(rate2_st*stst)-1*sum(rate3_st(:,:,i)*stst);';..
                ];
            else
                fst=[fst;..
                '      I(i)=sum(rate2_ts*stst)+sum(rate3_ts(:,:,i)*stst)+sum(rate3t_ts(:,:,i)*stst)-sum(rate2_st*stst)-sum(rate3_st(:,:,i)*stst)-sum(rate3t_st(:,:,i)*stst);';..
                ];
            end
            
        else
            fst=[fst;..
            '      I(i)=sum(rate2_ts*stst)-sum(rate2_st*stst);';..
            ];
        end
        fst=[fst;..
        '   end;';]
        if experiment.spectrum_deriv>0 then     //calculate first derivative
            fst=[fst;..
        '   y=(I(2:$)-I(1:$-1))/(dx*experiment.A);'];   
        elseif experiment.spectrum_deriv==0 then
           fst=[fst;..
        '   y=I/0.0129064/experiment.A;'];
        end
        
    else
        fst=[fst;..
        '//zero current approximation';..
        'if '+fst2+' calc.spec_raw==[] | c then';..
        '   x=%Data_trunc(:,1)-experiment.x0;'];
       
        if experiment.spectrum_deriv==2 then
            fst=[fst;..
            '   //2nd derivative';..
            '   dx=(x($)-x(1))/(length(x)-1);';..  //determine the differerence between datapoints 
            '   xx=[x;x($)+dx]-dx/2;'];            //expand the dataset by one point
        else
            fst=[fst;..
            '   xx=x;'];
        end
        
        if experiment.func_tip then
            fst=[fst;..
            '   //func_tip';..
            '   posbkup=experiment.position';..        //save the parameter page
            '   experiment.position=2;'];              //but always calculate the visible spin
        end

        fst=[fst;..
        '   experiment.xrange=xx;';..
        '   [xx,y]=spec2(experiment);'];        
    
        if experiment.third_order_calc then
            fst=[fst;..
            '   //3rd order calc';..
            '   occ=Occupation(experiment.Eigenval,experiment.T);'];
//            if experiment.entanglement then
//                fst=[fst;..
//                '//scattering also on the 2nd site';..
//                'for jp=1:size(experiment.atom,2),';..
//                '   experiment.jposition=jp;'];
//            else
            fst=[fst;.. //entanglement will be dealed on the M35 level
                '   experiment.jposition=experiment.position;'];
  //          end
            fst=[fst;..
            '   for i=1:size(experiment.Eigenvec,2),';..
            '       if occ(i)>%precission then';..
            '           [xx,y1,y1r]=spec3(experiment,i);';..
            '            y=y+y1+y1r;';..
            '       end,';..
            '   end'];
//            if experiment.entanglement then
//                fst=[fst;..
//                'end;'];
        end
    end

//    if experiment.func_tip then
//        fst=[fst;..
//        '   experiment.position=posbkup;'];  //playback the parameter page
//    end

    if experiment.spectrum_deriv==2 then
        fst=[fst;..
        '   y=(y(2:$)-y(1:$-1))/dx; //forward differentiation';];
    end

    fst=[fst;..
    '    calc.spec_raw=[x,y];';..
    'else';..
    '    x=calc.spec_raw(:,1);';..
    '    y=calc.spec_raw(:,2);';..
    'end;';..
    'fitres=[x,y];'];
    
    //calculate a scaling factor to get good fits under all conditions
    scale_f=size(%Data_trunc,1)/sum(abs(%Data_trunc(:,2))); 
                
    if isdef("%weight") then    //%weight allows to weight the data 
        fst=[fst;..
//        'fk=sum((%Data_trunc(:,2)-experiment.A*(real(y)+experiment.y0+experiment.b*%Data_trunc(:,1))).^2*%weight);'];
          'fk=sum(('+string(scale_f)+..
          '*(%Data_trunc(:,2)-experiment.A*(real(y)+experiment.y0+experiment.b*%Data_trunc(:,1)))).^2*%weight);'];
    else
        fst=[fst;..
        'fk=sum(('+string(scale_f)+..
        '*(%Data_trunc(:,2)-experiment.A*(real(y)+experiment.y0+experiment.b*%Data_trunc(:,1)))).^2);'];
    end

    fst=[fst;..
    'old_k=k;';..
    'disp(fk)';..
      //  'fk=y';..
    'endfunction']
endfunction
