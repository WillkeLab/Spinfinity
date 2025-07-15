////////////////////////////////////////////////////////////////////////////////
//
//    spin-spec-functions-1.9.5.sce
//
//    Stuttgart, Aachen, Jülich 04 April July 2021
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

global ggbr_table;

function gui_drawspec(optional)
    
  //if the optional parameter (independent what) is given, no questions are ask. 
      
  global main_gui guientry1 guislider1 gui experiment calc %stopcalc;
  
  [lhs,rhs]=argn(0);
    
  if experiment.Eigenval==[] then       //if we don't have the eigenvalues
      eigenvalues(experiment);          //and eigenvectors then calculate them!
  end;
  
  scf(main_gui);                        //switch to the correct graphic window
  
  nbr_of_atoms=size(experiment.atom,2);
  [nbr_max_states,nbr_of_states]=size(experiment.Eigenvec);
  maxcnt=nbr_of_states*nbr_of_atoms; 
  
  answ=[];
 
  if rhs==0 & max(experiment.Eigenval)-min(experiment.Eigenval)<max(experiment.xrange)..
      & nbr_of_states<nbr_max_states then
      msg=["The state energies are smaller than the bias range.",..
      "This might result in inaccurately calculated specta.",..
      "Increase ''max # of eigenstates'' under ''Parameters''"];
      answ=messagebox(msg,"modal","info", ["Continue", "Abort"]);
  end;
  
  if answ==2 then
      abort;
  end
  
  if calc.spec_raw==[] then         //no data so we have to newly calculate

     //branch off for different level of spectrum calculations

     select 1*experiment.rate_calc
         
     case 0 then
         disp("Zero I approx calculation")
         speccalc_zero_I_approx()
         
     case 1 then
         disp("Rate calculation")
         speccalc_with_rate()
         
     case 2 then
         disp("Symmetric rate calculation")
         speccalc_with_rate_symmetric()
         
     case 3 then
         disp("Bloch Redfield calculation")
         speccalc_with_bloch_redfield()
     end
  
  end  

  //from hereon all calculation brunches run together again

  if calc.spec==[] then
      if ~experiment.rate_calc then
          if %normalize then 
            atnr=experiment.position;
            scale=2*(experiment.atom(atnr).S*(experiment.atom(atnr).S+1)+experiment.atom(atnr).U^2)^-1;
          elseif ~experiment.use_T_ts then 
            scale=experiment.A;
          else
            scale=1;
          end;
      else      //calculated with rate equations or Bloch-Redfield approach
          if %normalize then 
            scale=1;
          else 
            scale=experiment.A;
          end;
      end;
      calc.spec=[calc.spec_raw(:,1)+experiment.x0];
      for i=2:size(calc.spec_raw,2),
          calc.spec=[calc.spec,(calc.spec_raw(:,i)+experiment.y0+experiment.b*calc.spec_raw(:,1))*scale];
      end
      //expand calc.spec by the average value for Bloch-Redfield calculations
      if experiment.rate_calc == 3 then
         time=[0;calc.spec(:,1)];
         dtime=time(2:$)-time(1:$-1);
         average=sum(calc.spec(:,2).*dtime)/sum(dtime);
         //average=sum(calc.spec(:,2))/length(calc.spec(:,2));
         calc.spec(:,3)=ones(length(calc.spec(:,2)))*average;
      end
  end;
 
  scf(main_gui);
  xmin=min(calc.spec(:,1));
  xmax=max(calc.spec(:,1));
  ymin=min(calc.spec(:,2:$));
  ymax=max(calc.spec(:,2:$));

//  plot2d(calc.spec(:,1),calc.spec(:,2:$),strf="011",rect=[xmin,ymin,xmax,ymax],frameflag=1);
  plot2d(calc.spec(:,1),calc.spec(:,2:$));
  
  select experiment.rate_calc
  case 0 then
      //no rate equations
     str='Spin #:'+string(nbr_of_atoms-size(calc.spec,2)+2:nbr_of_atoms);
     xlabeltext="eV (meV)";
     select experiment.spectrum_deriv
     case 1 then
         ylabeltext="dI/dV (a.u.)";
     case 2 then
         ylabeltext="d2I/dV2 (a.u.)";
     end
  
  case 1 then
     //rate equations
     str='Spin #:'+string(experiment.position);
     xlabeltext="eV (meV)";
     select experiment.spectrum_deriv
     case 0 then
         ylabeltext="I (nA)";
     case 1 then
         ylabeltext="dI/dV (quantum of conductance)";
     case 2 then
         ylabeltext="d2I/dV2 (quantum of conductance / mV)";
     end

  case 2 then
     //symmetric rate equations
     str='Spin #:'+string(experiment.position);
     xlabeltext="eV (meV)";
     select experiment.spectrum_deriv
     case 0 then
         ylabeltext="I (nA)";
     case 1 then
         ylabeltext="dI/dV (quantum of conductance)";
     case 2 then
         ylabeltext="d2I/dV2 (quantum of conductance / mV)";
     end
  case 3 then
      //Bloch-Redfield calculation
      str='Spin #:'+string(experiment.position);
      xlabeltext="time (ps)";
      select experiment.spectrum_deriv
      case 0 then
          ylabeltext="I (nA)";
      case 1 then
          ylabeltext="dI/dt (nA/ps)";
      case 2 then
          ylabeltext="d2I/dt2 (nA/(ps^2)";
      end  
      average=sum(calc.spec(:,2))/length(calc.spec(:,2));
      str=[str,"Average="+string(average)];
      
  end
  
  captions(gui.graph_ui.children(1).children,str);
  
  if experiment.rate_calc==3 then
      gui.graph_ui.children(2).children(1).line_style=3
  end
  
  gui.graph_ui.x_label.text=xlabeltext;
  gui.graph_ui.y_label.text=ylabeltext;

  select experiment.spectrum_deriv
  case 0 then       
      rnk=find(abs(calc.rank)>1);   //search for critical values with |rank| >1
      rnk2=find(calc.rank<0);      //search for critical values with rank <0
  case 1 then
      rnk=find((abs(calc.rank(1:$-1))+abs(calc.rank(2:$)))>2); //consider neighbors
      rnk2=find((calc.rank(1:$-1)<0)|(calc.rank(2:$)<0))    //consider neighbors
  case 2 then
      rnk=find((abs(calc.rank(1:$-2))+abs(calc.rank(2:$-1))+..
                abs(calc.rank(3:$)))>3); //consider nn and next nearest neighbors
      rnk2=find((calc.rank(1:$-2)<0)|(calc.rank(2:$-1)<0)|..
                (calc.rank(3:$)<0))    //consider nn and next nearest neighbors
  end

  if rnk<>[] then
      x=calc.spec(rnk,1);
      y=calc.spec(rnk,2);
      plot(x,y,"y.");
  end

  if rnk2<>[] then
      x=calc.spec(rnk2,1);
      y=calc.spec(rnk2,2);
      plot(x,y,"r.");
  end

endfunction

function speccalc_zero_I_approx()

    //////////////////////////////////////
    //zero current approximation calculation
    /////////////////////////////////////

    global main_gui guientry1 guislider1 gui experiment calc %stopcalc;
         
    cnt=0;
    winH=waitbar('Calculating Spectrum');
    
    daty=[];  //in this variable we store the "y" data, that is the calculated dI/dV
    
    occ=Occupation(experiment.Eigenval,experiment.T);
    
    if experiment.func_tip & size(experiment.atom,2)>1 then
        firstatnr=2;        // The tip spin is always atom number 1. 
                            // This spectrum we don't calculate directly.
                            // Therefore we start at firstatom=2.
    else
        firstatnr=1;
    end
    nbr_of_atoms=size(experiment.atom,2);
    
    gui_startcalc();
    
    for atnr=firstatnr:nbr_of_atoms,            //loop over all atomic sites
        experiment.position=atnr;               //set tip's position
        experiment.jposition=experiment.position; //and the sample-sample scattering position
    
        [x,y]=spec2(experiment); // second order calculation
        if %savedata then   
            csvWrite([x real(y)],"data_2nd"+string(atnr)+".dat"," "); 
        end;
    
        y2=0;
        
        // third order calculation
        if experiment.third_order_calc then
            for i=1:size(experiment.Eigenvec,2),    //loop over all initial states
//only consider processes which starts from states which have a minimal occupation    
                if occ(i) > %precission then
                    if ~%stopcalc then  //catch the stop button state
                         waitbar(cnt/maxcnt,['Calculating Spectrum';..
                        ' ';'Evaluate State '+string(i)+..
                        ' with tip at atom '+string(atnr)],winH);
                    else
                        close(winH);
                        gui_stopcalc(); disp("Interrupted calculation!")
                        abort;
                    end
                    [x,y1,y1r]=spec3(experiment,i); //calculate 3rd order contributions 
                   y2=y2+(y1+y1r); //sum them up in y2
                end,
                cnt=cnt+1;
            end;
            
            if %savedata then
                csvWrite([x real(y2)],"data_inelastic"+string(atnr)+".dat"," "); 
            end;
        end;
        scf(main_gui);
        
        daty=[daty,y+y2];   //put all data into "daty"

        if %savedata then
            csvWrite([x real(y+y2)],"data_inelastic_all"+string(atnr)+".dat"," ");
        end;
    end;        //loop end for calculating over all atomic sites
    
    gui_stopcalc("OK");
    close(winH);
   
    if experiment.spectrum_deriv == 2 then
        //calculate 2nd derivative from 1st derivative
        di=(daty(1:$-1,:)-daty(2:$,:))*experiment.spec_length/(2*experiment.xrange);
        xx=x-0.5*experiment.xrange/(experiment.spec_length-1);
        xx=xx(1:$-1);
        calc.spec_raw=[xx,di];
    else
        calc.spec_raw=[x,daty];
    end

endfunction

   
function speccalc_with_rate()
     
    global main_gui guientry1 guislider1 gui experiment calc %stopcalc;
      
      
    //////////////////////////////////////////////////////////////////
    //rate calculation 
    //The following scattering channels are accounted for:
    // (1) s->t 
    // (2) t->s
    // (3) s->s
    //
    //if the tip is "active" then also
    //
    // (4) t->t
    //////////////////////////////////////////////////////////////////
                        
      
        
    J2_ts=experiment.A;     //experiment.A is J^2 of the tip-sample junction
    spec_length=experiment.spec_length; //length of the spectrom
    ve=real(experiment.Eigenval);
    nbr_of_atoms=size(experiment.atom,2);
    [nbr_max_states,nbr_of_states]=size(experiment.Eigenvec);
 
    //calculate delta_E between the different states
    ediff=ones(1:size(ve,1))'.*.ve'-ones(1:size(ve,1)).*.ve;

    //initialize output variables
    y=zeros(spec_length,nbr_of_states);
    I=zeros(spec_length,1);
    ltime=zeros(spec_length,nbr_of_states);
    
    if experiment.func_tip & size(experiment.atom,2)>1 & experiment.position==1 then   
        experiment.position=2;    //the first spin with functionalized tip is 
                                  //"dark" which we do not calculate
        experiment.jposition=2;
    end
    
    //calculate the absolute square of the 2nd order tip_sample transition 
    //matrix elements "rate_ts=$|M_{t\rightarrow s}|^2$ with
    //$M_{t\rightarrow s}=\langle\psi_f|s\cdot S|\psi_i\rangle$ 
    
    if ~experiment.func_tip then
        rate_ts=Rate2nd2(experiment);     //normal transition rate calculation
    elseif experiment.func_tip==1 then
        //calculate the transition rate matrix with a functionalized (dark) tip 
        rate_ts=Rate2nd2_func_tip(experiment);
    else
        //calculate the transition rate matrix with a functionalized (bright) tip 
        rate_ts=Rate2nd2_func_tip_2(experiment);
    end

    rate_st=rate_ts';

    //set 'tip' equal to sample to calculate absolute square of the 2nd order 
    //sample-sample scattering matrix element "rate_ss=|<f|M|i>|^2"

    tempexperiment=experiment;
    
    tempexperiment.ptip=experiment.psample;         
    tempexperiment.ptipK=experiment.psampleK;
    tempexperiment.atom(:).U=0; //a U value would be canceled here, therefore we set it to zero
    
    //calculate 2nd order sample_sample transition matrix elements "rate_ss"
    if experiment.use_T_ts then
        rate_ss=Rate2nd2(tempexperiment); //the scaling is done in Rate2nd2 (version 1.6))        
    else
        rate_ss=experiment.atom(experiment.position).J^2*Rate2nd2(tempexperiment); 
    end
        
    // ATTENTION !
    // 
    // In principle we should calculate rate_ss for processes on all atoms, 
    // and not only the one where we calculate the spectrum, but this will 
    // lead to unavoidable problems (missing deconstruction of entanglements)
    // which can not be handled in this model. 
    // ("two weakly coupled spins problem")
    // We would need a more complex treatment via Bloch-Redfield approach
    
    // If you want to do this summation -- which might fail -- put the switch
    // experiment.entanglement to %T. 
    // However, I should discuss the consequences in more depth...
    
    
    if experiment.entanglement & nbr_of_atoms > 1 then
        atms=[1:nbr_of_atoms];
        atms(experiment.position)=[]; //we sum over all atoms except the one 
                                      //we are measuring

        if find(atms==1) & 1*experiment.func_tip>0 then
           functip=%T; //set marker for functionalized tip
           atms=atms(2:$); //remove the first element (i.e. the func. tip);
                           //we do this because the spin on the tip has to be 
                           //treated separately
        else
           functip=%F;
        end

        for atnr=atms,
            tempexperiment.position=atnr;
            if experiment.use_T_ts then
                rate_ss=rate_ss+Rate2nd2(tempexperiment); //new (version 1.6)
            else
                rate_ss=rate_ss+experiment.atom(atnr).J^2*Rate2nd2(tempexperiment); 
            end
        end
        
        if functip==%T then     //now we treat the scattering in the tip's spin:
           tempexperiment.position=1;   //set spin position to the one in the tip 
           //set 'sample' equal to tip to calculate tip-tip scattering
           tempexperiment.ptip=experiment.ptip;         
           tempexperiment.psample=experiment.ptip;         
           tempexperiment.ptipK=experiment.ptipK;
           tempexperiment.psampleK=experiment.ptipK;
           if experiment.use_T_ts then
              rate_ss=rate_ss+Rate2nd2(tempexperiment); //new (version 1.6)
           else
              rate_ss=rate_ss+experiment.atom(1).J^2*Rate2nd2(tempexperiment); 
           end
       end
        
    end
    
     //calculate 2nd order sample_sample transition rates "rate2_ss" by scaling 
     //the matrix elements with the energy difference between the states.
    rate2_ss=(fbox((ediff)/experiment.T)*experiment.T).*rate_ss;
    
    gui_startcalc();
    
    if experiment.third_order_calc then
        
        experiment.jposition=experiment.position;
        [rate3_st,rate3_ts]=spec3rate(experiment);  //calculate the 3rd order 

        //contribution to the rate_st and rate_ts

        rate3_ss=spec3rate_ss(experiment);  //new (version 1.6) the J is now included
        
        if experiment.entanglement & nbr_of_atoms > 1 then
            atms=[1:nbr_of_atoms];
            atms(experiment.position)=[]; //we sum over all atoms except the one we are measuring

            if find(atms==1) & 1*experiment.func_tip>0 then
                functip=%T; //set marker for functionalized tip
                atms=atms(2:$); //remove the first element (i.e. the func. tip)
            else
                functip=%F;
            end

            bckup_position=experiment.position;
            for atnr=atms,
                 experiment.position=atnr;
                 experiment.jposition=atnr;
                 rate3_ss=rate3_ss+spec3rate_ss(experiment) //new (version 1.6)
            end
            experiment.position=bckup_position;
            experiment.jposition=bckup_position;
            
            if functip==%T then
                tempexperiment=experiment;
                tempexperiment.position=1;   //tip position
                //set 'sample' equal to tip to calculate tip-tip scattering
                tempexperiment.psample=experiment.ptip;         
                tempexperiment.psampleK=tempexperiment.ptipK;

                rate3_ss=rate3_ss+spec3rate_ss(experiment) //new (version 1.6)
            end
  
        end

        if ~experiment.use_T_ts then    //new in version 1.6
            rate3_ts=rate3_ts*J2_ts;
            rate3_st=rate3_st*J2_ts;
        end

    end;
    
    if %stopcalc then
        gui_stopcalc();
        abort;
    end
    i=0;
    x=linspace(experiment.xrange,-experiment.xrange,spec_length);
    
    winH=waitbar('Calculating Spectrum (Searching for the Matrix Kernel)');
    for U=x,
        i=i+1;
        
        // calculate 2nd order sample_tip (tip_sample) transition rates 
        // "rate2_ts" and "rate2_st" by scaling the matrix elements with 
        // the energy difference between the states PLUS or MINUS the applied bias.

        rate2_ts=J2_ts*fbox((ediff+U)/experiment.T)*experiment.T.*rate_ts;
        rate2_st=J2_ts*fbox((ediff-U)/experiment.T)*experiment.T.*rate_st;
        
        if experiment.third_order_calc then    
            //sum up all transition rates in 2nd and 3rd order
            rate2=rate2_ss+rate3_ss+rate2_ts+rate2_st+rate3_ts(:,:,i)+..
            rate3_st(:,:,i);
        else
            //sum up all transition rates in 2nd order
            rate2=rate2_ss+rate2_ts+rate2_st;
        end;
        
        //write down the maxtrix of differential equations: 
        // whereby we have rate2= 
        //$\begin{pmatrix} \Gamma_{1\rightarrow 1} & \Gamma_{2\rightarrow 1} & \cdots & \Gamma_{n\rightarrow 1} \\ \Gamma_{1\rightarrow 2} & \Gamma_{2\rightarrow 2}& \cdots & \Gamma_{n\rightarrow 2} \\ \vdots & \vdots & \ddots & \vdots \\ \Gamma_{1\rightarrow n} & \Gamma_{2\rightarrow n} & \cdots & \Gamma_{n\rightarrow n} \end{pmatrix}$
        //with the following clever trick we rewrite it in a set of differential
        //equations of the following form:
        //$0=-p_1\sum_1^n\Gamma_{1\rightarrow n} +p_2\Gamma_{2\rightarrow 1}+p_3\Gamma_{3\rightarrow 1}+\dots+p_n\Gamma_{n\rightarrow 1}$
        //where we are searching the $p_n$
        
        rate21=rate2-eye(rate2).*rate2-diag(sum(rate2-eye(rate2).*rate2,1));
        
        [ststsol]=kernel(rate21); //find steady-state solution
        
        if %record_trans_matrix then
            calc.trans_matrix=[calc.trans_matrix,matrix(rate21,-1,1)];
        end
        
        
//  To always find the best solution we will now loop over all possible solutions
//  and chose the one which is the most physical one

        j=0;
        while j<size(ststsol,2);    
            j=j+1;
            sc=sum(ststsol,1);    
            stst=ststsol(:,j)/sc(j);  //normalize to get occupancy of the states
        
            if max(stst)>1+%eps | min(stst)<-0-%eps then
                calc.rank(i)=-size(ststsol,2);
                //save the rank of the matrix i.e. the # of possible solutions
                //invert the rank to mark that the occupation is out of the 
                //[0,1] range. This is evidently a gross error 
                //and therefore the result is later plotted in "RED"
            else
                calc.rank(i)=size(ststsol,2); 
                //save the rank of the matrix, i.e. the # of possible solutions
                break;  //it is "normal" solution therefore stop the search.
            end
        end

        y(i,:)=stst';   //y stores the state occupations
        
        //the inverse of the lifetime is:
        ltime(i,:)=sum(rate21-rate21.*eye(rate21),1);
           
        
        // current is the sum of the state occupations times transition 
        // probabilities evaluated for current flowing from tip->sample and 
        // from sample->tip
        
        if  experiment.third_order_calc then
            I(i)=sum(rate2_ts*stst)-sum(rate2_st*stst)+..
            sum(rate3_ts(:,:,i)*stst)-sum(rate3_st(:,:,i)*stst);
            Itot(i)=sum(rate2*stst);
        else
            I(i)=sum(rate2_ts*stst)-sum(rate2_st*stst);
            Itot(i)=sum(rate2*stst);
        end;
        if ~%stopcalc then
            waitbar(i/spec_length,'Calculating Spectrum (searching the matrix kernel)',winH);
        else
            close(winH);
            gui_stopcalc();
            abort;
        end

    end; 

    gui_stopcalc("OK");
    close(winH);
    
    select experiment.spectrum_deriv
    case 0 then
        I=I/(experiment.A*0.0129064) //rescale to nA
        calc.spec_raw=[x',I];
    case 1 then
        // first derivative dI/dV
        di=(I(1:$-1)-I(2:$))*(spec_length-1)/(2*experiment.xrange);
        xx=x-experiment.xrange/(spec_length-1);
        xx=xx(1:$-1);
        di=di/experiment.A;
        calc.spec_raw=[xx',di];
    case 2 then
        // second derivative dI^2/d2V
        di=(I(1:$-2)-2*I(2:$-1)+I(3:$))*spec_length*(spec_length-1)/(4*experiment.xrange^2);
        xx=x(2:$-1);
        di=di/experiment.A;
        calc.spec_raw=[xx',di];
    end

    calc.occ=[x',y];
    calc.lifetime=[x',ltime.^-1*6.582e-13*2]; //
    calc.Itot=[x',Itot];
     
      
  endfunction
  
  function speccalc_with_rate_symmetric()

  //////////////////////////////////////////////////////////////////
  //rate calculation with all scattering in sample as well as in tip
  //////////////////////////////////////////////////////////////////
  //
  //
  //The following scattering channels are accounted for:
  // (1) s->t 
  // (2) t->s
  // (3) s->s
  // (4) t->t
  //
  // an active tip spin is not accounted for, neither correlations
  // in the baths induced by the spins
  //
  //////////////////////////////////////////////////////////////////
                        
  
  global main_gui guientry1 guislider1 gui experiment calc %stopcalc;
   

    J2_t=experiment.A;
    J_t=-sqrt(experiment.A);
    spec_length=experiment.spec_length; //length of the spectrom
    ve=real(experiment.Eigenval);   
 
    //calculate delta_E between the different states
    ediff=ones(1:size(ve,1))'.*.ve'-ones(1:size(ve,1)).*.ve;

    //initialize output variables
    y=zeros(spec_length,nbr_of_states);
    I=zeros(spec_length,1);
    ltime=zeros(spec_length,nbr_of_states);
    
    if experiment.func_tip then
        disp("Functionalized tip not supported in this transport mode");
        messagebox("Functionalized tip not supported in this transport mode", "Warning", "warning");
        abort;
    end
    
    //calculate 2nd order tip_sample transition matrix elements "rate_ts"
    rate_ts=Rate2nd2(experiment);     //normal transition rate calculation

    //The 2nd order sample_tip transition matrix elements "rate_st" are the transpose of "rate_ts" 
    rate_st=rate_ts';   

    
    //set tip equal to sample to calculate absolute square of the 2nd order 
    //sample-sample scattering matrix element "rate_ss=|<f|M|i>|^2"

    tempexperiment=experiment;
    tempexperiment.ptip=experiment.psample;         
    tempexperiment.atom(:).U=0; //a U value would be canceled here, therefore we set it to zero
     
    //calculate 2nd order sample_sample transition matrix elements "rate_ss"
    if experiment.use_T_ts then
        disp("The use of T_ts is not supported in this transport mode");
        messagebox("The use of T_ts is not supported in this transport mode", "Warning", "warning");
        abort;
    end
    
    J2_s=experiment.atom(experiment.position).J^2;
    rate_ss=J2_s*Rate2nd2(tempexperiment); 

    // ATTENTION !
    // 
    // In principle we should calculate rate_ss for processes on all atoms, 
    // and not only the one where we calculate the spectrum, but this will 
    // lead to unavoidable problems (missing deconstruction of entanglements)
    // which can not be handled in this model. 
    // ("two weakly coupled spins problem")
    // I guess we would need a more complex treatment via Bloch-Redfield
    
    // If you want to do this summation -- which might fail -- put the switch
    // experiment.entanglement to %T. 
    // However, I should discuss the consequences in more depth...
    
    
    if experiment.entanglement & nbr_of_atoms > 1 then
        atms=[1:nbr_of_atoms];
        atms(experiment.position)=[]; //we sum over all atoms except the one we are measuring

        for atnr=atms,
            tempexperiment.position=atnr;

            rate_ss=rate_ss+experiment.atom(atnr).J^2*Rate2nd2(tempexperiment); 
        end
    end

    //calculate 2nd order tip_tip transition matrix elements "rate_tt"        
    //set 'sample' equal to tip to calculate tip-tip scattering
    tempexperiment.ptip=experiment.ptip;
    tempexperiment.psample=experiment.ptip;
    tempexperiment.position=experiment.position;         
    rate_tt=J2_t*Rate2nd2(tempexperiment); 
    
     //calculate 2nd order sample_sample transition rates "rate2_ss" scaling 
     //the matrix elements with the energy difference between the states.
     //and do the same for "rate2_tt"
     
    rate2_ss=(fbox((ediff)/experiment.T)*experiment.T).*rate_ss;
    rate2_tt=(fbox((ediff)/experiment.T)*experiment.T).*rate_tt;
    
    gui_startcalc();
    
    J2_ts=(J2_s*J2_t)/(J2_s+J2_t); //calculate the coupling between tip and sample
    //using a simple Resistor-Resistor model!
    
    if experiment.third_order_calc then
        
        // calculate 3rd order s_ts components

        experiment.jposition=experiment.position;
        [rate3_st,rate3_ts]=spec3rate(experiment);
        
        // calculate 3rd order t_ts components 

        tempexperiment=experiment;
        tempexperiment.ptip=experiment.psample;         
        tempexperiment.psample=experiment.ptip;         
        tempexperiment.entanglement=%f;
        tempexperiment.atom(experiment.position).J=J_t;
        tempexperiment.xrange=-experiment.xrange;
        [rate3t_ts,rate3t_st]=spec3rate(tempexperiment);
//        rate3t_st=rate3t_st(:,:,$:-1:1);    //invert the bias direction
//        rate3t_ts=rate3t_ts(:,:,$:-1:1);
       
       // calculate 3rd order s_ss components 
       
        rate3_ss=spec3rate_ss(experiment);  //new (version 1.6) the J are now included
        
        if experiment.entanglement & nbr_of_atoms > 1 then
            atms=[1:nbr_of_atoms];
            atms(experiment.position)=[]; //we sum over all atoms except the one we are measuring

            bckup_position=experiment.position;
            for atnr=atms,
                 experiment.position=atnr;
                 experiment.jposition=atnr;
                 rate3_ss=rate3_ss+spec3rate_ss(experiment) //new (version 1.6)
            end
            experiment.position=bckup_position;
            experiment.jposition=bckup_position;
            
        end

        rate3_tt=spec3rate_tt(experiment);  //new (version 1.8) 

        //scale all 3rd order ts rates with the J2_ts coupling between tip and sample
        rate3_ts=rate3_ts*J2_ts;
        rate3_st=rate3_st*J2_ts;
        rate3t_ts=rate3t_ts*J2_ts;
        rate3t_st=rate3t_st*J2_ts;

    end;
    
    if %stopcalc then
        gui_stopcalc();
        abort;
    end
    i=0;
    x=linspace(experiment.xrange,-experiment.xrange,spec_length);
    
    winH=waitbar('Calculating Spectrum (searching the matrix kernel)');
    for U=x,
        i=i+1;
        
        // calculate 2nd order sample_tip (tip_sample) transition rates 
        // "rate2_ts" and "rate2_st" by scaling the matrix elements with 
        // the energy difference between the states.

        rate2_ts=J2_ts*fbox((ediff+U)/experiment.T)*experiment.T.*rate_ts;
        rate2_st=J2_ts*fbox((ediff-U)/experiment.T)*experiment.T.*rate_st;
        
        if experiment.third_order_calc then    
            //sum up all transition rates in 2nd and 3rd oder
            rate2=rate2_ss+rate2_tt+rate3_ss+rate3_tt+rate2_ts+rate2_st+rate3_ts(:,:,i)+..
            rate3_st(:,:,i)+rate3t_ts(:,:,i)+rate3t_st(:,:,i);
            
        else
            //sum up all transition rates in 2nd oder
            rate2=rate2_ss+rate2_tt+rate2_ts+rate2_st;
        end;
        
        //write down the maxtrix of differential equations: 
        rate21=rate2-eye(rate2).*rate2-diag(sum(rate2-eye(rate2).*rate2,1));
        [ststsol]=kernel(rate21); //find steady-state solution
        
        if %record_trans_matrix then
            calc.trans_matrix=[calc.trans_matrix,matrix(rate21,-1,1)];
        end
//        
//        if size(ststsol,2)>1 then
//            messagebox("Rank of matrix smaller than dimension!..
//            Calculation of kernel fails.", "Error", "error");
//            close(winH);
//            gui_stopcalc();
//            experiment.rate2_ts=rate2_ts;
//            experiment.rate2_st=rate2_st;
//            experiment.rate2_ss=rate2_ss;
//            experiment.rate21=rate21;
//            experiment.ststsol=ststsol;
//            experiment.errU=U;
//            experiment.erry=y;
//            experiment.errlt=ltime;
//            experiment.errI=I;
//            abort;
//        end;    

//  To always find the best solution we will now loop over all possible solutions
//  and chose the one which is the most physical one

        j=0;
        while j<size(ststsol,2);    
            j=j+1;
            sc=sum(ststsol,1);    
            stst=ststsol(:,j)/sc(j);  //normalize to get occupancy of the states
        
            if max(stst)>1+%eps | min(stst)<-0-%eps then
                calc.rank(i)=-size(ststsol,2); 
                //invert the rank to mark that the occupation is out of the [0,1] range
                //this is evidently a gross error 
                //and therefore the result is later plotted in "RED"
            else
                calc.rank(i)=size(ststsol,2); 
                break;
            end
        end

        y(i,:)=stst';   //y stores the state occupations
        
        //the inverse of the lifetime is:
        ltime(i,:)=sum(rate21-rate21.*eye(rate21),1);
           
        
        // current is the sum of the state occupations times transition 
        // probabilities evaluated for current flowing from tip->sample and 
        // from sample->tip
        
        if  experiment.third_order_calc then
            I(i)=sum(rate2_ts*stst)-sum(rate2_st*stst)+..
            sum(rate3_ts(:,:,i)*stst)+sum(rate3t_ts(:,:,i)*stst)-sum(rate3_st(:,:,i)*stst)-sum(rate3t_st(:,:,i)*stst);
//            sum(rate3_ts(:,:,i)*stst)-sum(rate3_st(:,:,i)*stst);
            Itot(i)=sum(rate2*stst);
        else
            I(i)=sum(rate2_ts*stst)-sum(rate2_st*stst);
            Itot(i)=sum(rate2*stst);
        end;
        if ~%stopcalc then
            waitbar(i/spec_length,'Calculating Spectrum (searching the matrix kernel)',winH);
        else
            close(winH);
            gui_stopcalc();
            abort;
        end

    end; 

    gui_stopcalc("OK");
    close(winH);
    
    select experiment.spectrum_deriv
    case 0 then
        I=I/(experiment.A*0.0129064) //rescale to nA
        calc.spec_raw=[x',I];
    case 1 then
        // first derivative dI/dV
        di=(I(1:$-1)-I(2:$))*(spec_length-1)/(2*experiment.xrange);
        xx=x-experiment.xrange/(spec_length-1);
        xx=xx(1:$-1);
        di=di/experiment.A;
        calc.spec_raw=[xx',di];
    case 2 then
        // second derivative dI^2/d2V
        di=(I(1:$-2)-2*I(2:$-1)+I(3:$))*spec_length*(spec_length-1)/(4*experiment.xrange^2);
        xx=x(2:$-1);
        di=di/experiment.A;
        calc.spec_raw=[xx',di];
    end

    calc.occ=[x',y];
    calc.lifetime=[x',ltime.^-1*6.582e-13*2]; //scaling factor to make lifetime [s]
    //the odd number 
    calc.Itot=[x',Itot];

endfunction
  
function speccalc_with_bloch_redfield(varargin)
    
  //calculate the steady state density-matrix:
  
  global main_gui guientry1 guislider1 gui experiment calc %stopcalc;
  
  [lhs,rhs]=argn(0);
  
  //experiment.position=gui.sel_spin.value;//we place the tip on the selected site
  
  //check if experiment.RedF_dm_calc_list and experiment.RedF_dm_init are
  // correctly initialized
  
  if experiment.RedF_dm_init==[] then       //if not defined assume thermal equilibrium
      ggbr_button_reset_thermal();
  end;
  
  if experiment.RedF_dm_calc_list==[] then
      gui_Bloch_Redfield();
      messagebox(["Bloch-Refield parameters are missing!";..
      "Please initialze!"],"warning");
      abort;
  end

  
  xrange_backup=experiment.xrange;
  dm=experiment.RedF_dm_init;
  experiment.RedF_coeff=[];
  spec_length=length(experiment.RedF_dm_calc_list(:,1));
  calc.dm=zeros(size(dm,1),size(dm,1),spec_length);
//  calc.spec=zeros(spec_length,2);
  I=zeros(spec_length,1);
  calc.occ=zeros(spec_length,size(experiment.Eigenvec,2)+1);
  calc.lifetime=zeros(spec_length,size(experiment.Eigenvec,2)+1);
  Time=zeros(spec_length,1);
  lst=experiment.RedF_dm_calc_list;
  i=1;
  t_offset=0;

  if rhs==0 then
    gui_startcalc();
    winH=waitbar('Calculating Evolution of the Density Matrix and Spectrum');
    sleep(100); 
  end

  
  while i<spec_length+1,

      if rhs==0 then
          waitbar(i/spec_length,winH);      
          if %stopcalc then
              gui_stopcalc();
              close(winH)
              abort;
          end
      end
     
      t=lst(i,1);
      for j=i+1:spec_length,
          if lst(j,2)==lst(i,2) then
             t=[t,lst(j,1)];
          else
              j=j-1;
              break;
          end
      end
      if j==[] then
          j=i;
      end
            
      experiment.xrange=lst(i,2);
      [calc.dm(:,:,i:j),I(i:j),occ,ltime,trans_matrix,calc.rank(i:j)]=Redfield_dynamic2(experiment,dm,t-t_offset);
      Time(i:j)=t;
      calc.occ(i:j,1)=t;
      calc.occ(i:j,2:$)=occ;
      calc.lifetime(i:j,1)=t;
      calc.lifetime(i:j,2:$)=(ltime.^-1*6.582e-13*2)'.*.ones(t)';
      t_offset=t($);
      dm=calc.dm(:,:,j);
      i=j+1;
      if %record_trans_matrix then
          calc.trans_matrix=[calc.trans_matrix;trans_matrix];
      end
      
      
  end
    
  experiment.xrange=xrange_backup;
 
  if rhs==0 then
    gui_stopcalc("OK");
    close(winH);
  end

  
  time_range=experiment.RedF_dm_calc_list($,1);
  
  select experiment.spectrum_deriv
  case 0 then
      I=I/(experiment.A*0.0129064); //rescale to nA
      calc.spec_raw=[Time,I];
  case 1 then
      // first derivative dI/dt
      di=(I(1:$-1)-I(2:$))//*spec_length/time_range;
      dt=(Time(1:$-1)-Time(2:$));
      xx=(Time(1:$-1)+Time(2:$))/2;
      di=(di./dt)/(experiment.A*0.0129064); //rescale to nA/ps
      calc.spec_raw=[xx,di];
  case 2 then
      // second derivative dI^2/d2V
      di=(I(1:$-2)-2*I(2:$-1)+I(3:$))//*spec_length*(spec_length-1)/(time_range^2);
      dt=(Time(1:$-2)-2*Time(2:$-1)+Time(3:$));
      xx=(Time(1:$-2)+2*Time(2:$-1)+Time(3:$))/4;
      di=(di./dt)/(experiment.A*0.0129064); //rescale to nA/ps^2;
      calc.spec_raw=[xx,di];
  end
  
  ggbr_update();
  show_ini_dm_update();
      
endfunction
  
