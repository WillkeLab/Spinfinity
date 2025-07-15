////////////////////////////////////////////////////////////////////////////////
//
//    Spin-Gui-1.9.5 sce
//
//    Stuttgart, Aachen, Jülich 08 October 2021
//
//    (C) Markus Ternes
//  
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by5
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
// 
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more detai5ls.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
////////////////////////////////////////////////////////////////////////////////


clear();
clearglobal();
PATH=get_absolute_file_path('Spin-Gui-1.9.5_mod.sce');
PATHi=PATH; //PATHi stays the initialisation path for saving status

helperfile="Spin-Hamiltonians-1.9.5.sce";
ierr=execstr('exec(PATH+helperfile,-1)','errcatch');  //load some usefull functions
if ierr<>0 then 
    disp('file '+helperfile+' missing!'); 
    abort;
end

helperfile="Spin-additional-functions-1.9.5.sce"; 
ierr=execstr('exec(PATH+helperfile,-1)','errcatch');  //load some usefull functions
if ierr<>0 then 
    disp('file '+helperfile+' missing!'); 
    abort;
end

helperfile="Spin-spec-functions-1.9.5.sce"; 
ierr=execstr('exec(PATH+helperfile,-1)','errcatch');  //load some usefull functions
if ierr<>0 then 
    disp('file '+helperfile+' missing!'); 
    abort;
end

global %Bloch_Redfield;
helperfile="Spin-BR-1.9.5.sce"; 
ierr=execstr('exec(PATH+helperfile,-1)','errcatch');  //load some usefull functions
if ierr<>0 then 
    disp('file '+helperfile+' missing!'); 
    %Bloch_Redfield=%F; 
else
    %Bloch_Redfield=%T;
end

helperfile="Spin-fit-functions-1.9.5.sce";
ierr=execstr('exec(PATH+helperfile,-1)','errcatch');  //load some usefull functions
if ierr<>0 then 
    disp('file '+helperfile+' missing! Fitting disabled'); 
end

helperfile="IBM_QNS_import.sce";
ierr=execstr('exec(PATH+helperfile,-1)','errcatch');  //load some usefull functions
if ierr<>0 then 
    disp('file '+helperfile+' missing!'); 
end

//global %guidebug          //switch on for debugging of the GUI

global PATH;
global PATHi;

global %savedata;
global %plotspec;
global %precission;
global %normalize;
global %zeroadj;
global %stopcalc;
global %renormalize_E;

global %Data;        //experimental Data
global %Data_trunc;  //experimental truncated Data
global Data_fname;   //experimental Data filename
global calc;         //calculated results
global old_k;       //old fitting parameters
global itcount;     //fitting iteration counter
global minvalues1;  //minium of slider and fit range
global maxvalues1;  //maxium of slider and fit range
global labels1;     //Slider lables 
global %Stevstr;    //Stevenoperator lables
global scrpt;        //script text

global guicheckbox1;//gui elements
global guientry1; 
global guislider1;
global guitext1;
global gui;
global gg ggbr;  //buttons in tip/sample gui
global main_gui ts_gui cm_gui ani_gui br_gui;   //graphic handles

clear("experiment");
clear("experiment2");

global experiment;  //main parameterfile
global experiment2;

global gui_last_spin; 

global Examples_73 Actions_73;

ierr=execstr('stacksize(''max'');','errcatch');  //increase stack for scilab 5.x.x
ierr=execstr('stacksize(2.237D+08);','errcatch');

// define an atom

atomlink.S=0.5;             //spin
atomlink.g=2;               //g-factor                     
atomlink.D=0;               //axial anisotropy
atomlink.E=0;               //inplane anisotropy
atomlink.alpha=0;           //extrinsic rotation around x-axis (version 1.7)
atomlink.beta=0;            //extrinsic rotation around y-axis (version 1.7)
atomlink.gamma=0;           //extrinsic rotation around z-axis (version 1.7)
atomlink.J=-0.04;           //Jrho_0 coupling to electrode
atomlink.U=0;               //Coulomb scattering parameter
atomlink.w=20;              //bandwidth 
atomlink.use_Stevens=0;     //use D/E anisotropy parameters
atomlink.Stevens=[0];

//define the experiment

experiment.T=1*%kBoltzmann; //Temperature in meV
experiment.xrange=10;       //bias range from -10mV to +10mV
experiment.lt=0.005;        //intrinsic lifetime broadening 5 ueV
experiment.ptip=[0 0 0];    //spin polarisation in the tip electrode
experiment.psample=[0 0 0]; //spin polarisation in the sample electrode
experiment.ptipK=[];
experiment.psampleK=[];
experiment.atom=atomlink;
experiment.Eigenvec=[];
experiment.Eigenval=[];
experiment.position=1;      //tip's position on extendet spin systems
experiment.jposition=1;     //3rd order scattering position with sample bath
experiment.A=1;             //amplitude of the spectrum; when 
                            //experiment.rate_calc=%T then this parameter 
                            // is the channel transparency
                            
experiment.T_ts=[1 1 1];    //tip_sample transfer parameter; used only when 
                            //experiment.use_T_ts=%T then tip sample scattering 
                            //is sum_{i=x,y,z}(T_ts_{i}*sigma_i*S_{i}
                            //new in version 1.6
                            
experiment.b=0;             //additional background slope for spectrum
experiment.x0=0;            //bias offset
experiment.y0=0;            //spectrum offset
experiment.B=[0 0 4];       //magnetic field
experiment.matrix='1';      //Heisenberg interaction matrix between spins
experiment.matrixDM='0, 0, 0';  //DM interaction matrix between spins
experiment.Phase='0';         //Phase-factor between spins
experiment.heisenberg_coupling=0;   //scaling factor for spin-spin interactions
experiment.sef=1;           //scaling factor for spin-electrode correlations
                            //only used when experiment.func_tip>0 AND
                            //experiment.corr_sample=%T OR experiment.corr_tip=%T
                            
experiment.paramagnetic=%F; //if set then tip is paramagnetic and experiment.ptip
                            //is calculated via the Brillouin-function
experiment.paramag_S=5/2;   //spin of paramagnetic tip
experiment.paramag_g=2;     //g-factor of paramagnetic tip
experiment.eta=0.3;         //saturation polarisation of paramagnetic tip
experiment.no_eval=1000;    //number of calculated states 
experiment.spec_length=1000;//length of calculated spectrum (new in version 1.6)
experiment.max_no_eigenstates=50;
experiment.third_order_calc=%T; //if %F then spectrum is only calculated in second order
experiment.rate_calc=%F;    //if %T then rate equations are used for spectrum
                            //if %F then the zero-current approximation is used 
experiment.entanglement=%F; //if %T then 3rd oder scattering can occur on all spins
experiment.allatomsequ=0;   //if %T then all parameters of a set of atoms are simultaneously changed
experiment.func_tip=0;      //if 0: standart not functionalized tip
                            //if 1; functionalized tip but spectroscopically dark
                            //if 2: functionalized spectroscopically active tip
experiment.corr_sample=%F;  //correlate sample polarization with sample spin
experiment.corr_tip=%F;     //correlate tip polarization with tip spin
experiment.spectrum_deriv=1;//0: I(V), 1: dI/dV(V), 2: dI^2/d^2V(V)
experiment.use_T_ts=%F;     //new in version 1.6


if %Bloch_Redfield then
    experiment.RedF_t_total=100000;
    experiment.RedF_V_pulses=[0,5000];
    experiment.RedF_V_amplitudes=10;
    experiment.RedF_V_Gauss_Noise=0;
    experiment.RedF_MW_pulses=[];
    experiment.RedF_MW_amplitudes=0;
    experiment.RedF_V_func="";

    experiment.RedF_secular_window=0.01;
    experiment.RedF_secular="window";
    experiment.RedF_dm_init=[]; //initialization matrix for Bloch-Redfield calculation
    experiment.RedF_dm_calc_list=[];
    experiment.RedF_base="energy";
end



calc.spec = [];
calc.states= [];
calc.occ = [];
calc.lifetime = [];
calc.entropy = [];
calc.giantspin= [];
calc.negativity= [];
calc.magnetization=[];      //new in version 1.9.1
calc.correlations=[];       //new in version 1.9.1
calc.QM_correlations=[];    //new in version 1.9.1
calc.average=[];            //new in version 1.9.3
calc.multidata=[];          //new in version 1.9.3
calc.T2=[];                 //new in version 1.9.4
calc.spec_raw=[];
calc.rank=[];
calc.Itot=[];           //experimental!!!!
calc.trans_matrix=[];       //new in version 1.9.4

experiment2=experiment;

%savedata =%F; 
// If %savedata = %T all intermediate calculated data is saved
// The files are saved in the standart directory
// and contain files like: 
// data_(in)_(mid)_(fin).dat for 3rd order (normal order)
// data_(in)_(mid)_(fin)r.dat for 3rd order (reversed order)
// data_(in)_(fin)SF.dat for 2nd order processes;

%plotspec =%F;
// if %plotspec = %F then all intermediate states are plotted

%precission=0.001;
// %precission sets the mininal (in) state occupation to be calculated
// and %precission^2 the minimal matrix element to be evaluated
// the calculation speed drastically depends on that value

%normalize=%F;
// all spectra are normalized to 'one' at high voltage

%zeroadj=%T;
// the ground state is set to zero when plotting states

%renormalize_E=%F;
// do not renormaize the energies 
// (see: Cohen-Tannoudji, C., Dupont-Roc, J. & Grynberg, G. 
// "Atom-Photon Interaction" Wiley and Sons, Inc., New York, 1989)

%record_trans_matrix=%T;
// record the transition matrix elements in rate-equation and Bloch-redfield
// calculations 


%stopcalc=%F;

gui_last_spin=1;


/////////////////////////////////////////
// build the gui:
/////////////////////////////////////////

function build_main_gui()

    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;
    
    global experiment guicheckbox1 guientry1 guislider1 gui main_gui guitext1;
    global maxvalues1 minvalues1 labels1 checkboxvalues1 %Stevstr;
    global Examples_73 Actions_73 gui_last_spin;
       
    use_Stevens=experiment.atom(gui_last_spin).use_Stevens;
    %Stevstr="";
    Stevclr=zeros(3,7);
    if 1*use_Stevens>0 then
       maxS=max(experiment.atom(:).S);
       %Stevstr="$B_2^0$";
       Stevclr(1,1)=1;
       if use_Stevens==1 then
          %Stevstr=[%Stevstr,"$B_2^1$"];
          Stevclr(1,2)=1;
       end
       if use_Stevens<3 then
          %Stevstr=[%Stevstr,"$B_2^2$"];
          Stevclr(1,3)=1;
       end
       if maxS>1.5 then
          %Stevstr=[%Stevstr,"$B_4^0$"];
          Stevclr(2,1)=1;
          if use_Stevens==1 then
            %Stevstr=[%Stevstr,"$B_4^1$"];
            Stevclr(2,2)=1;
          end
          if use_Stevens<3 then
            %Stevstr=[%Stevstr,"$B_4^2$"];
            Stevclr(2,3)=1;
          end
          if use_Stevens==3 | use_Stevens==1 then
            %Stevstr=[%Stevstr,"$B_4^3$"];
            Stevclr(2,4)=1;
          end
          if use_Stevens==1 |use_Stevens==2 | use_Stevens==4 then
            %Stevstr=[%Stevstr,"$B_4^4$"];
            Stevclr(2,5)=1;
          end
       end
       if maxS>2.5 & use_Stevens>1 then   
           //only when there is at least a bit of symmetry
           
          %Stevstr=[%Stevstr,"$B_6^0$"];
          Stevclr(3,1)=1;
          if use_Stevens==2 then
            %Stevstr=[%Stevstr,"$B_6^2$"];
            Stevclr(3,3)=1;
          end
          if use_Stevens==3 then
            %Stevstr=[%Stevstr,"$B_6^3$"];
            Stevclr(3,4)=1;
          end
          if use_Stevens==2 | use_Stevens==4 then
            %Stevstr=[%Stevstr,"$B_6^4$"];
            Stevclr(3,5)=1;
          end
          if use_Stevens==3 | use_Stevens==5 then
            %Stevstr=[%Stevstr,"$B_6^6$"];
            Stevclr(3,7)=1;
          end
       end
    end
    
    //clear parameters which do not match symmetry
//    for i=1:size(experiment.atom,2), 
//        experiment.atom(i).Stevens=experiment.atom(i).Stevens.*Stevclr;
//    end

    experiment.atom(gui_last_spin).Stevens=experiment.atom(gui_last_spin).Stevens.*Stevclr;

        
    // positioning
    alx = 20; aly = 495+20; alyd = 20;
    extray=max(size(%Stevstr,2)-2,0)*alyd;
    aly=aly+extray;
    ierr = execstr("posbk=main_gui.position;delete(main_gui)", "errcatch");
    sleep(100);
    
    main_gui=figure('figure_id',73,'figure_position',[1000,0],..
    'figure_size',[1250,760+extray],..
    'auto_resize','on','background',[33],'figure_name',..
    '3rd order tunneling simulator. (c) 2010-2021 Markus Ternes, Version 1.9.5 EXPERIMENTAL',..
    'resize','on',..
    'closerequestfcn',"gui_exit()");
    if ierr==0 then
        main_gui.position=posbk;
    end 


    drawlater();
    
    delmenu(main_gui.figure_id,gettext('?'));
    toolbar(main_gui.figure_id,'on');
    
    gui.user_data = [];
    gui.graph_ui= newaxes();
    gui.graph_ui.margins = [0.001 0.001 0.001 0.001];
    gui.graph_ui.axes_bounds = [0.37,0.03,0.62,0.85];
    
    // Frames creation 
    frame_1 = uicontrol("parent",main_gui, "relief","groove", ..
    "style","frame", "units","pixels", ..
    "position",[ 10, 10, 380, 620+ extray], ..
    "horizontalalignment","center", "background",[0.9 0.9 0.9], ..
    "tag","frame_control");
    
    uicontrol("parent",main_gui, "style","text",..
    "string","$S$", "position",[alx,aly,28,20], ..
    "horizontalalignment","center", "fontsize",12, ..
    "TooltipString", "Total spin of the spin system", ..
    "background",[0.9 0.9 0.9]);
    
    gui.S = uicontrol("parent",main_gui, "style","edit",..
    "string","0.5", "position",[alx+30,aly,50,20], ..
    "horizontalalignment","center", "fontsize",11, ..
    "background",[1 1 1], "tag","spin",..
    "TooltipString", "Total spin of the spin system", ..
    "Callback", 'entry1_callback()');
    
    gui.spin_nr = uicontrol("parent",main_gui, "style","text",..
    "string","Sample Spin #", "position",[alx+210,aly+2,90,20], ..
    "horizontalalignment","left", "fontsize",11, ..
    "TooltipString", "Select spin system", ..
    "background",[0.9 0.9 0.9]);
    
    gui.sel_spin = uicontrol("parent",main_gui, "style","spinner",..
    "position",[alx+300,aly,50,25], ..
    "horizontalalignment","center", "fontsize",11,'Max',[100],'Min',[1],..
    'SliderStep',[1,1],"background",[1 1 1], "tag",'sel_spin',..
    "TooltipString", "Select spin system", ..
    "fontsize",12,..
    "String", "test", ..
    "Tag", "sel_spin", ..
    "Value", string(gui_last_spin), ..
    "Callback","slider_spin_sel_callback()");
   
    if use_Stevens==0 then
       // ordered list of labels
       labels1 = ["$g$", "$D$", "$E$", "$\alpha$", "$\beta$", "$\gamma$",..
       "$U$", "$J\rho_s$", "$\omega_0$","$T_{0}^2$", "$b$", "$V_{off}$",..
       "$\sigma_0$","$T_{eff}$","$\eta_{tip}$",..
       "$B_Z$","$B_Y$","$B_X$","$V_R$"];
       values1 = [2,0,0,0,0,0,0,0,20,1,0,0,0,1,0,4,0,0,10];
       maxvalues1 = [4, 10, 10, %pi, %pi, %pi, 1, 0.0, 200, 10, 0.05, 5, 5, 10, 1, 20, 20, 20, 200];
       minvalues1 = [0, -10, -10, -%pi, -%pi, -%pi, -1, -1, 2, 0, -0.05, -5, -1, 0.1, -1, -20, -20, -20, 2];
       checkboxvalues1=[0,0,0,1,1,1,0,0,1,0,1,1,1,0,1,0,0,0,0];
       ttlabels1 = ["g-factor of the spin",..
       "Axial magnetic anisotropy (meV)",..
       "In-plane magnetic anisotropy (meV)",..
       "Extrinsic rotation around x-axis (rad)",..
       "Extrinsic rotation around y-axis (rad)",..
       "Extrinsic rotation around z-axis (rad)",..
       "Coulomb scattering term",..
       "Kondo scattering term",..
       "Sample electron bandwidth (meV)",..
       "Tip-sample interaction strength",..
       "Additional background slope",..
       "Additional voltage offset (mV)",..
       "Additional background conductance",..
       "Effective temperature (K)",..
       "Spin polarization in the tip",..
       "Magnetic field strength along Z (T)",..
       "Magnetic field strength along Y (T)",..
       "Magnetic field strength along X (T)",..
       "Bias range (mV)"];
    else
       labels1 = ["$g$",%Stevstr, "$\alpha$", "$\beta$", "$\gamma$", "$U$", "$J\rho_s$", "$\omega_0$",..
       "$T_{0}^2$", "$b$", "$V_{off}$","$\sigma_0$","$T_{eff}$","$\eta_{tip}$",..
       "$B_Z$","$B_Y$","$B_X$","$V_R$"];
        values1 = [2, zeros(1,size(%Stevstr,2)), 0, 0, 0, 0, 0, 20, 1, 0, 0, 0, 1, 0, 4, 0, 0, 10];
        maxvalues1 = [4, ones(1,size(%Stevstr,2)), %pi, %pi, %pi,..
         1, 0.0, 200, 10, 0.05, 5, 1, 10, 1, 20, 20, 20, 200];
        minvalues1 = [0, -ones(1,size(%Stevstr,2)),-%pi, -%pi, -%pi..
         -1, -1, 2, 0, -0.05, -5, -1, 0.1, -1, -20, -20, -20, 2];
        checkboxvalues1=[0,zeros(1,size(%Stevstr,2)),1,1,0,0,1,0,1,1,1,0,1,0,0,0,0];
        ttlabels1 = ["g-factor of the spin"];

        for i=1:size(%Stevstr,2),
            ttlabels1 = [ttlabels1,"Magnetic anisotropy parameter"];
        end

        ttlabels1 = [ttlabels1,..
       "Extrinsic rotation around x-axis (rad)",..
       "Extrinsic rotation around y-axis (rad)",..
       "Extrinsic rotation around z-axis (rad)",..
        "Coulomb scattering term",..
        "Kondo scattering term",..
        "Sample electron bandwidth (meV)",..
        "Tip-sample interaction strength",..
        "Additional background slope",..
        "Additional voltage offset (mV)",..
        "Additional background conductance",..
        "Effective temperature (K)",..
        "Spin polarization in the tip",..
        "Magnetic field strength along Z (T)",..
        "Magnetic field strength along Y (T)",..
        "Magnetic field strength along X (T)",..
        "Bias range (mV)"];
    end
    
   // delete(guicheckbox1);
   // delete(guientry1);
   // delete(guislider1);

    if %Stevstr<>"" then 
        maxsl=size(%Stevstr,2)+13;
    else
        maxsl=15;
    end

    for k=1:length(values1),
        if labels1(k)=="$g$" then...........//new since Version 1.9.4
            guitext1(k) = uicontrol("parent",main_gui, "style","pushbutton", ..
            "Position",[alx,aly-k*alyd,28,20], ..
            "BackgroundColor",[0.9,0.9,0.9],..
            "horizontalalignment","center",..
            "string",labels1(k),..
            "Relief","flat",..
            "TooltipString", ttlabels1(k), ..
            "Callback","gui_g_change()");
        elseif labels1(k)=="$J\rho_s$" then     //new since Version 1.6
            guitext1(k) = uicontrol("parent",main_gui, "style","pushbutton", ..
            "Position",[alx,aly-k*alyd,28,20], ..
            "BackgroundColor",[0.9,0.9,0.9],..
            "horizontalalignment","center",..
            "string",labels1(k),..
            "Relief","flat",..
            "TooltipString", ttlabels1(k), ..
            "Callback","gui_Js_change()");
        elseif labels1(k)=="$\eta_{tip}$" then
            guitext1(k) = uicontrol("parent",main_gui, "style","pushbutton", ..
            "Position",[alx,aly-k*alyd,28,20], ..
            "BackgroundColor",[0.9,0.9,0.9],..
            "horizontalalignment","center",..
            "string",labels1(k),..
            "Relief","flat",..
            "TooltipString", ttlabels1(k), ..
            "Callback","gui_ts_change()");
        elseif labels1(k)=="$T_{0}^2$" & experiment.use_T_ts then  //new in version 1.6
            labels1(k)="$T_{ts}$";  
            guitext1(k) = uicontrol("parent",main_gui, "style","pushbutton", ..
            "Position",[alx,aly-k*alyd,28,20], ..
            "BackgroundColor",[0.9,0.9,0.9],..
            "horizontalalignment","center",..
            "string",labels1(k),..
            "Relief","flat",..
            "TooltipString", ttlabels1(k), ..
            "Callback","gui_T_ts_change()");
        elseif labels1(k)=="$U$" & experiment.use_T_ts  //new in version 1.6
            guitext1(k) = uicontrol("parent",main_gui, "style","text",..
            "string","$U_{ts}$", "position",[alx,aly-k*alyd,28,20], ..
            "horizontalalignment","center", "fontsize",12, ..
            "TooltipString", ttlabels1(k), ..
            "background",[1 1 0.6]);
        else
            guitext1(k) = uicontrol("parent",main_gui, "style","text",..
            "string",labels1(k), "position",[alx,aly-k*alyd,28,20], ..
            "horizontalalignment","center", "fontsize",12, ..
            "TooltipString", ttlabels1(k), ..
            "background",[0.9 0.9 0.9]);
        end
            
        guientry1(k) = uicontrol("parent",main_gui, "style","edit",..
        "string",string(values1(k)), "position",[alx+240,aly-k*alyd,85,20], ..
        "horizontalalignment","center", "fontsize",11, ..
        'Max',1,'Min',0,"TooltipString", ttlabels1(k), ..
        "background",[1 1 1], "tag",labels1(k),"Callback",..
        'entry1_callback()');
        guislider1(k) = uicontrol("parent",main_gui, "style","slider",..
        "string",string(values1(k)), "position",[alx+35,aly-k*alyd+3,200,15], ..
        "horizontalalignment","center", "fontsize",11,'Max',[100],'Min',[0],..
        'SliderStep',[1],"background",[1 1 1], "tag",labels1(k),..
        "TooltipString", ttlabels1(k), ..
        "Callback",'slider1_callback()');
        
        if k<=maxsl then
            guicheckbox1(k) = uicontrol("parent",main_gui, "style","checkbox",..
            "string","", "position",[alx+335,aly-k*alyd,20,20], ..
            "horizontalalignment","center", "fontsize",11,'Value',checkboxvalues1(k),..
            'SliderStep',[1,10],"background",[0.9 0.9 0.9],..
            "TooltipString", "Disable variable from fitting", ..
            "Callback", 'checkbox1_callback()');
        end
    end
    
    uicontrol("parent",main_gui, "style","text",..
    "string","# of Spins", "position",[alx,aly+85,150,20], ..
    "horizontalalignment","left", "fontsize",12, ..
    "TooltipString", "Total number of coupled spin systems", ..
    "background",[0.9 0.9 0.9]);
    
    gui.nbr_spins = uicontrol("parent",main_gui, "style","spinner",..
    "string","1", "position",[alx+60,aly+80,60,27], ..
    "horizontalalignment","center", "fontsize",11,'Max',[100],'Min',[1],..
    "background",[1 1 1], "tag","nbr_spins",..
    'SliderStep',[1,1],..
    "TooltipString", "Total number of coupled spin systems", ..
    "Callback", 'number_spins_callback()');
    
    uicontrol("parent",main_gui, "style","text",..
    "string","$\mathcal{C}$", "position",[alx,aly+55,28,20], ..
    "horizontalalignment","center", "fontsize",12, ..
    "TooltipString", "Interaction scaling parameter (AFM:+  FM:-)", ..
    "background",[0.9 0.9 0.9]);
    
    gui.sel_heisenberg = uicontrol("parent",main_gui, "style","edit",..
    "string","1", "position",[alx+240,aly+55,85,20], ..
    "horizontalalignment","center", "fontsize",11, ..
    "background",[1 1 1], "tag","sel_heisenberg", ..
    "TooltipString", "Interaction scaling parameter (AFM:+  FM:-)", ..
    "Callback",'entry1_callback();');
    
    gui.sel_heisenberg2 = uicontrol("parent",main_gui, "style","slider",..
    "string","1", "position",[alx+35,aly+58,200,15], ..
    "horizontalalignment","center", "fontsize",11,'Max',[100],'Min',[0],..
    'SliderStep',[1],"background",[1 1 1], "tag",'sel_heisenberg2',..
    "TooltipString", "Interaction scaling parameter (AFM:+  FM:-)", ..
    "Callback",'slider1_callback()');
    
     uicontrol("parent",main_gui, "style","text",..
    "string","$i\infty j$", "position",[alx+160,aly,30,20], ..
    "TooltipString", "Enable 3rd order scattering also with other spin systems", ..
    "horizontalalignment","left", "fontsize",12, ..
    "background",[0.9 0.9 0.9]);
    
    guicheckbox1(maxsl+1) = uicontrol("parent",main_gui,..
    "style","checkbox","string","", "position",[alx+335,aly+55,20,20], ..
    "horizontalalignment","center", "fontsize",11,'Value',1,..
    'SliderStep',[1,10],"background",[0.9 0.9 0.9], ..
    "TooltipString", "Disable variable from fitting", ..
    "Callback",'checkbox1_callback()');
    
    gui.entanglement = uicontrol("parent",main_gui, "style","checkbox",..
    "string","", "position",[alx+140,aly,20,20],..
    "horizontalalignment","center", "fontsize",11,'Value',0,..
    "background",[0.9 0.9 0.9], ..
    "TooltipString", "Enable 3rd order scattering also with other spin systems",..
     "Callback",'gui_setmode();');
  
    // buttonsize and pos
    blx = 100; bly = 25; bx=30;by=15;
        
    gui.mode1 = uicontrol("parent",main_gui, "Position", [bx,aly+25,blx+5,bly],..
    "Style", "popupmenu",..
    "TooltipString", "Select physical model for the calculation",..
    "BackgroundColor",[0.6,0.9,0.9], "fontsize",11, ..
    "Relief","raised",..
    "String", "2nd Order|2nd+Rate|2nd+SymRate|Bloch-Redfield|3rd Order|3rd+Rate|3rd+SymRate",..// "4th Order",..
    "Callback", "gui_setmode();",..
    "value", 5*experiment.third_order_calc+1*experiment.rate_calc,..
    "Tag", "change_mode_button");

    gui.spectrum_deriv = uicontrol("parent",main_gui, "Position", [bx+120,aly+25,blx,bly],..
    "Style", "popupmenu",..
    "TooltipString", "Select derivative of the calculation",..
    "BackgroundColor",[0.6,0.9,0.9], "fontsize",11,..
    "Relief","raised",..
    "String", "I(V)|dI/dV(V)|d²I/dV²(V)",..
    "Callback", "gui_setmode();",..
    "value", experiment.spectrum_deriv+1,..
    "Tag", "change_spec_deriv_button");

    // Adding button
    gui.button_del_all = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+0.55*blx by 0.45*blx bly], "String","Del All", ..
    "BackgroundColor",[1,0.6,0.5], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Delete all graphical elements in main Graph", ..
    "Callback","gui_del_all()");
    
    gui.button_del = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx by 0.45*blx bly], "String","Del", ..
    "BackgroundColor",[1,0.6,0.5], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Delete the last graphical element in main Graph", ..
    "Callback","gui_del()");
     
    gui.button_fit = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+120 by blx*0.45 bly], "String","Fit", ..
    "BackgroundColor",[0.3,0.9,0.3], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Open submenu to fit experimental data", ..
    "Callback","gui_fit()");
    
    gui.button_limits = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+120+0.55*blx by blx*0.45 bly], "String","Limits", ..
    "BackgroundColor",[0.3,0.9,0.3], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Open submenu to set parameter limits", ..
    "Callback","gui_limits();");
    
    
    gui.button_draw_spec = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+120 by+90 blx bly], "String","Draw Spectrum", ..
    "BackgroundColor",[0.7,0.9,0.7], "fontsize",11, ..
    "TooltipString", "Calculate the spectrum / spectra versus bias", ..
    "Relief","raised",..
    "Callback","gui_drawspec()");
    
    gui.button_draw_states = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+120 by+60 blx bly], "String","Draw States", ..
    "BackgroundColor",[0.7,0.9,0.7], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Calculate the state energies versus magnetic field", ..
    "Callback","gui_drawstates()");
    
    gui.button_anisotropy = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+120 by+30 blx bly], "String","Draw Anisotropy", ..
    "BackgroundColor",[0.7,0.9,0.7], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Change the symmetry and calculate anisotropy energy landscape for the active spin", ..
    "Callback","gui_checkanisotropy();");
    
    gui.button_stopcalc = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+0.55*blx by+30 0.45*blx bly], "String","$\blacksquare$", ..
    "BackgroundColor",[0.6,0.6,0.9], "fontsize",11, ..
    "Relief","raised",..
    "foregroundcolor",[0.8 0.8 0.8],..
    "enable","off",..
    "TooltipString", "Stop the calculation", ..
    "callback_type", 10, ..
    "Callback","gui_stopcalc(1)");
    
     gui.button_clear = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx by+30 0.45*blx bly], "String","Clr Calc", ..
    "BackgroundColor",[1,0.6,0.5], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Clear all calculations and results", ..
    "Callback","cleardata(2)");
    
    gui.button_eigenstates = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx by+90 blx bly], "String","Further Results", ..
    "BackgroundColor",[0.9,0.9,0.5], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Open submenu to plot further calculated data", ..
    "Callback","gui_eigenstates()");
    
    gui.button_switchexp = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+240+0.55*blx by+90 blx*0.45 bly], "String","$1\Leftrightarrow 2$", ..
    "BackgroundColor",[0.9,0.9,0.5], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Toggle the two experimental parameter sets", ..
    "Callback","gui_switchexp()");
    
    gui.button_multiplot = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+240 by+90 blx*0.45 bly], "String","Script", ..
    "BackgroundColor",[0.7,0.9,0.7], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Multiple plots generated by simple scripts", ..
    "Callback","scrpt=gui_script(scrpt)");
    
    gui.button_load = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+240 by+30 blx bly], "String","Load", ..
    "BackgroundColor",[0.7,0.7,0.9], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Load experimental data or program settings", ..
    "Callback","gui_load()");
    
    uicontrol("parent",main_gui, "style","text",..
    "string","Column #:", "position",[bx+240,by+55,70,20], ..
    "horizontalalignment","left", "fontsize",12, ..
    "TooltipString", "Number of the column containg the ordinate (I, dI/dV or dI^2/d^V) data", ..
    "background",[0.9 0.9 0.9]);
    
    gui.sel_Load_Column = uicontrol("parent",main_gui, "style","edit",..
    "string","3", "position",[bx+300,by+55,40,20], ..
    "horizontalalignment","center", "fontsize",11, ..
    "TooltipString", "Number of the column containg the ordinate (I, dI/dV or dI^2/d^V) data", ..
    "background",[1 1 1], "tag",'sel_Load_Column');
    
    gui.button_save = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+240 by blx bly], "String","Save", ..
    "BackgroundColor",[0.7,0.7,0.9], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Save calculated data or program settings", ..
    "Callback","gui_save()");
    
    gui.button_tipsample = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+240 aly+80 blx bly], "String","Tip/Sample", ..
    "BackgroundColor",[0.6,0.9,0.9], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Open submenu to change tip and sample spin polarization", ..
    "Callback","gui_tipsample()");
    
    gui.button_couplingmatrix = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+120 aly+80 blx bly], "String","Coupling Matrix", ..
    "BackgroundColor",[0.6,0.9,0.9], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Open submenu to change interactions beween the spin systems", ..
    "Callback","gui_couplingmatrix()");
    
    gui.button_parameters = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+240 aly+25 blx*0.6 bly], "String","Params", ..
    "BackgroundColor",[0.7,0.7,0.9], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Open submenu to change global parameters", ..
    "Callback","gui_parameters()");

    gui.button_help = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx+240+blx*0.7 aly+25 blx*0.3 bly], "String","?", ..
    "BackgroundColor",[0.9,0.9,0.5], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Open help", ..
    "Callback","gui_help()");

    gui.button_bloch_redfield = uicontrol("parent",main_gui, "style","pushbutton", ..
    "Position",[bx by+60 blx bly], "String","Bloch-Redfield", ..
    "BackgroundColor",[0.6,0.9,0.9], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Open submenu to change global parameters", ..
    "Callback","gui_Bloch_Redfield();");
    
    gui.allatomsequ = uicontrol("parent",main_gui, "style","checkbox",..
    "string","", "position",[alx+80,aly,20,20], ..
    "horizontalalignment","center", "fontsize",11,'Value',0,..
    "background",[0.9 0.9 0.9], ..
    "TooltipString", "Change parameters on all spin systems simultaneously", ..
    "Callback",'gui_setmode();');
    
    uicontrol("parent",main_gui, "style","text",..
    "string","$\ddagger$", "position",[alx+100,aly,20,20], ..
    "horizontalalignment","left", "fontsize",12, ..
    "TooltipString", "Change parameters on all spin systems simultaneously", ..
    "background",[0.9 0.9 0.9]);

    drawnow();
    
    //disp("Addmenu")
    
    addmenu(main_gui.figure_id,'Actions',..
    ['Draw &Spectrum';'Draw S&tates';'Draw &Anisotropy';..
    'More Results'; 'Edit Tip/Sample';..
    'Edit Spin-Spin Couplings';'Edit Parameters']);
    Actions_73=['gui_drawspec()';'gui_drawstates()';'gui_checkanisotropy()';..
    'gui_eigenstates()';'gui_tipsample()';'gui_couplingmatrix()';..
    'gui_parameters()'];
 
    
//    addmenu(main_gui.figure_id,'Actions',..
//    ['Draw Spectrum'],list(2,"gui_drawspec(experiment)"));
    
    addmenu(main_gui.figure_id,'Examples',..
    ['Spin-1/2';'Spin-1';'Fe/CuN';'Mn/CuN';'Co/CuN';'NiOc/Cu(111)']);
    tmptxt="=[''S.S=0.5;S.g=1.97;S.D=0;S.E=0;S.U=0;S.J=-0.04;S.w=20;udata(S)'';";
    tmptxt=tmptxt+..
    "''S.S=1;S.g=2;S.D=-5;S.E=1;S.U=0.1;S.J=-0.1;S.w=20;udata(S)'';";
    tmptxt=tmptxt+..
    "''S.S=2;S.g=2.11;S.D=-1.57;S.E=0.31;S.U=0.35;S.J=-0.087;S.w=20;udata(S)'';";
    tmptxt=tmptxt+..
    "''S.S=5/2;S.g=1.9;S.D=-0.051;S.E=0;S.U=1.3;S.J=-0.029;S.w=20;udata(S)'';";
    tmptxt=tmptxt+..
    "''S.S=3/2;S.g=2;S.D=2.7;S.E=0.5;S.U=0;S.J=-0.25;S.w=20;udata(S)'';";
     tmptxt=tmptxt+..
    "''S.S=1;S.g=2;S.D=3.5;S.E=0.0;S.U=0.08;S.J=-0.08;S.w=20;udata(S)'';";
    str="Examples_"+string(main_gui.figure_id)+tmptxt+"];";
    execstr(str);
    //disp(ierr);
    
    update_gui_values(); 
    checkbox1_callback(); 
    number_spins_callback(); // initialize

endfunction

////////////////////////////////////////////////////////////////////////////////
// callback functions
////////////////////////////////////////////////////////////////////////////////

function gui_startcalc()
    
    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;

    global gui %stopcalc;
    gui.button_stopcalc.enable='on';
    gui.button_stopcalc.foregroundcolor=[1 0.1 0.1];
    %stopcalc=%F; //reset %stopcalc flag
    
endfunction
    
function gui_stopcalc(varargin)
    
    //Depending on the parameter "varargin", the stopbutton and the global 
    //variable %stopcalc is set. This subroutine allows to intersect calculations
    //
    //if varargin=="" then button is set gay, all data is deleted, and %stopcalc=%T
    //this shall be called when the calculation is aborted
    //
    //if varargin=="OK" then button is set gray, %stopcalc=%f.
    //this shall be called to initialize the button and variable when a 
    //calculation finished without interruption
    //
    //if varargin=="trigger" then button is set yellow, %stopcalc=%T.
    //this shall be called to alarm the system to interupt the calculation ASAP

    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;

    global gui %stopcalc;
    [lhs,rhs]=argn(0); 
    gui.button_stopcalc.enable='off';
    %stopcalc=%T;    //set the %stopcalc flag
    
    if rhs==0 then
        gui.button_stopcalc.foregroundcolor=[0.8 0.8 0.8]; //gray
        cleardata(2); //delete all;
    else
        if varargin(1)=="OK" then
            gui.button_stopcalc.foregroundcolor=[0.8 0.8 0.8]; //gray
            %stopcalc=%f; //reset %stopcalc flag
           //all o.K
        else,
            gui.button_stopcalc.foregroundcolor=[1 1 0.1]; //yellow
            //interruption
        end;
    end;
    
endfunction

function gui_disable()
    
    //disable slider, buttons, and entryfields in the GUI 

    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;

    global guicheckbox1 guientry1 guislider1 gui;
    
    if %Stevstr<>"" then 
        maxsl=size(%Stevstr,2)+17;
    else
        maxsl=19;
    end
    
    guicheckbox1(1:maxsl-4).enable='off';
    guientry1(1:maxsl).enable='off';
    guislider1(1:maxsl).enable='off';
    gui.nbr_spins.enable='off';
    gui.sel_heisenberg.enable='off'
    gui.sel_heisenberg2.enable='off'
    gui.S.enable='off';
    
endfunction

function gui_enable()
    
    //enable the GUI to allow interaction with the user

    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;

    global guicheckbox1 guientry1 guislider1 gui experiment;
    
    if %Stevstr<>"" then 
        maxsl=size(%Stevstr,2)+17;
    else
        maxsl=19;
    end
    
    guientry1(1:maxsl).enable='on';
    guislider1(maxsl-3:maxsl).enable='on';  //enable B-field
    guicheckbox1(1:maxsl-4).enable='on';
    gui.nbr_spins.enable='on';
    gui.S.enable='on';
    checkbox1_callback()

endfunction

function cleardata(varargin)
    
    // depending on the parameter "varargin" the function clears data in the global 
    // variable "calc" and in "experiment.Eigenval" and "experiment.Eigenvec"
    //
    // if varargin=0 then only calc.spec is deleted
    // if varargin>=1 then all entries in calc.spec, except "giantspin" and "negativity"
    // if varargin>=2 all is deleted. This is the default if no parameter is given

    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
        disp(varargin)
    end;
    
    [lhs,rhs]=argn(0);
    if rhs==0 then 
        a=2; 
    else
        a=varargin(1);
    end
    
    global gui experiment calc %Bloch_Redfield;
    calc.spec=[];
    
    if a>=1 then
        calc.spec_raw=[];
        calc.states=[];
        calc.occ=[];
        calc.lifetime=[];
        calc.entropy=[];
        calc.rank=[];
        calc.dm=[];
        calc.magnetization=[];
        calc.correlations=[];
        calc.QM_correlations=[];
        calc.trans_matrix=[];
        experiment.RedF_coeff=[];
    end
    if a>=2 then
        experiment.Eigenval=[];
        experiment.Eigenvec=[];
        //experiment.RedF_dm_calc_list=[];
        experiment.RedF_dm_init=[];
        calc.giantspin=[];
        calc.negativity=[];
    end
    
    if %Bloch_Redfield then
        ggbr_update();
    end
    

endfunction

function checkbox1_callback()

    // this function is called when one of the checkboxes in the GUI is changed

    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;
    
    global guicheckbox1 guientry1 guislider1 gui main_gui experiment;

    if %Stevstr<>"" then 
        maxsl=size(%Stevstr,2)+13;
    else
        maxsl=15;
    end
    
    if experiment.paramagnetic then
       guicheckbox1(maxsl).Value=1; //switch off rho_tip for a paramagnetic tip
    end
    
//    for i= 1:length(guicheckbox1)-1     //last entry is Heisenberg Coupling. Treat seperately.

    for i= 1:maxsl
        if guicheckbox1(i).Value==0 then
            guientry1(i).enable="on";
            guislider1(i).enable="on";
            if experiment.func_tip & gui.sel_spin.value==1 & i<maxsl-5 then
                guientry1(i).backgroundcolor=[1 0.8 0.8];
            else
                guientry1(i).backgroundcolor=[1 1 1];
            end
            
            experiment.sw(i)=%T;
        else
            guientry1(i).enable="on";
            guislider1(i).enable="on";
            guientry1(i).backgroundcolor=[0.8 0.8 0.8];
            experiment.sw(i)=%F;
        end
    end
       
    if guicheckbox1(maxsl+1).Value==0 & size(experiment.atom,2)>1 then
        gui.sel_heisenberg.enable="on";
        gui.sel_heisenberg2.enable="on";
        gui.sel_heisenberg.backgroundcolor=[1 1 1];
        experiment.sw(maxsl+1)=%T;
    elseif size(experiment.atom,2)==1 then
        gui.sel_heisenberg.enable="off";
        gui.sel_heisenberg2.enable="off";
        guicheckbox1(maxsl+1).Value=1;
        experiment.sw(maxsl+1)=%F;
    else
        gui.sel_heisenberg.enable="on";
        gui.sel_heisenberg2.enable="on";
        gui.sel_heisenberg.backgroundcolor=[0.8 0.8 0.8];
        experiment.sw(maxsl+1)=%F;
    end

endfunction

function slider1_callback()

    // this function is called when a slider in the GUI is changed
    // after some checking it changes the values in "guientry1" and
    // calls the subroutine
    // update_atomlink()    to transfer the data in the GUI to the 
    // corrensponding entries in the global variable experiment

    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;

    global guicheckbox1 guientry1 guislider1 gui main_gui experiment;
    
    if %Stevstr<>"" then 
        maxsl=size(%Stevstr,2)+17;
    else
        maxsl=19;
    end
    
    for i= 1: maxsl,
         temp=guislider1(i).Value;
         maxi=maxvalues1(i);
         mini=minvalues1(i);
         r=maxi-mini;
         guientry1(i).string=string(clean(mini+r/100*temp));
    end


    // Heisenberg slider
    temp=gui.sel_heisenberg2.Value;
    gui.sel_heisenberg.string=string(-10+20/100*temp);
    
    update_atomlink();
    
endfunction

function entry1_callback()
        
    // this function is called when an entry in the GUI is changed
    // after some checking it calls the subroutines
    // update_slider()      to update the sliders in the gui
    // update_atomlink()    to transfer the data in the GUI to the 
    // corrensponding entries in the global variable experiment
    
    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;

    global guicheckbox1 guientry1 guislider1 gui experiment;
    
    if %Stevstr<>"" then 
        maxsl=size(%Stevstr,2)+17;
    else
        maxsl=19;
    end
    
    for i= 1:maxsl,
         guientry1(i).string(~isnum(guientry1(i).string))='0'; //replace bullshit
    end
    
    // Spin entry
    gui.S.string(~isnum(gui.S.string))='0'; //replace bullshit
    temp=round(evstr(gui.S.string)*2);
    if temp<1 then
        temp=1;
    elseif temp>20 then
        temp=20;
    end
    gui.S.String=string(temp/2);
    
    // Heisenberg entry
    gui.sel_heisenberg.String(~isnum(gui.sel_heisenberg.String))='0'; //replace bullshit
        
    update_slider();
    update_atomlink();
    
endfunction

function update_slider()
    
    // this function is called to update the sliders in the GUI when an
    // entry in guientry1 is changed
    
    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;

    
    global guientry1 guislider1 gui;
    
    if %Stevstr<>"" then 
        maxsl=size(%Stevstr,2)+17;
    else
        maxsl=19;
    end
    
    for i= 1:maxsl
         temp=evstr(guientry1(i).string);
         maxi=maxvalues1(i);
         mini=minvalues1(i);
         r=maxi-mini;
        guislider1(i).Value=(temp-mini)*100/r;
    end

    // Heisenberg entry
    temp=evstr(gui.sel_heisenberg.String);
    gui.sel_heisenberg2.Value=(temp+10)*100/20;
    
    //change color when parameters of the dark atom on the tip are diplayed
     
    if experiment.func_tip & gui.sel_spin.value==1 then
        guislider1(1:maxsl-10).backgroundcolor=[1 0.8 0.8];
    else
        guislider1(1:maxsl-10).backgroundcolor=[1 1 1];
    end
    
endfunction

function slider_spin_sel_callback()
    
    // this function is called when the actual spin in the GUI has
    // (possibly) changed.
    // it checks the correct range, updates if neccessary the Anisotropy figure
    // and triggers update_gui_vales
    
    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;
    
    global guicheckbox1 guientry1 guislider1 gui experiment gui_last_spin;
    sleep(100); //wait for settle down the Value
    
    sel=findobj('Tag', 'sel_spin');
    temp=sel.Value;

    if temp-floor(temp)>0 & temp-floor(temp)<0.5 then  //round down
        temp=floor(temp);
    elseif ceil(temp)-temp>0 & ceil(temp)-temp<0.5 then //round up
        temp=ceil(temp);
    else
        temp=round(temp);
    end
    
    if  temp<1 then
        temp=1;
    end
    
    if  temp>size(experiment.atom,2) then
        temp=size(experiment.atom,2);
    end
       sel.Value=temp;

// this part is no longer needed!       
//    if experiment.rate_calc then
//        cleardata(1); //spectral data is now invalid
//    end
    
    //check if figure ani_gui exist
    ierr = execstr("delete(ani_gui.children)", "errcatch"); 
    if ~ierr then
//       gui_disable();
       gui_checkanisotropy(); //yes then update figure
//       gui_enable();
    end
    
    if experiment.atom(gui_last_spin).use_Stevens <> experiment.atom(temp).use_Stevens then
        gui_last_spin=temp;
        build_main_gui();
    elseif gui_last_spin<>temp;
        update_gui_values();        
    end

    gui_last_spin=temp;

endfunction

function number_spins_callback()
    
    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;

    // this function is called when the total number of spins in the system has
    // (possibly) changed.
    // it expands or shrinks the size of experiment.atom according to the
    // value in gui.nbr_spins
    
    global guicheckbox1 guientry1 guislider1 gui experiment gui_last_spin;
    
    sleep(100); //wait for settle down the values in the GUI 

    temp=gui.nbr_spins.Value;   
    if temp-floor(temp)>0 & temp-floor(temp)<0.5 then  //round down
        gui.nbr_spins.Value=floor(temp);
    elseif ceil(temp)-temp>0 & ceil(temp)-temp<0.5 then //round up
        gui.nbr_spins.Value=ceil(temp);
    else
        gui.nbr_spins.Value=round(temp);
    end
    
    temp=round(gui.nbr_spins.value);
    
    if temp>15 then 
        temp=15;            //15 coupled spins are the maximum we allow
    end
    
    if temp<1 then 
        temp=1;
    end
        
    if temp<>gui.nbr_spins.value then
        gui.nbr_spins.value=temp;
    end;
    
    if or(%Stevstr<>"") then 
        maxsl=size(%Stevstr,2)+14;
    else
        maxsl=16; //the last entry is the Heiseberg coupling
    end
    
    if temp==1 then         //only one spin-system
        temp=1;
        gui.sel_spin.Enable="off";  //no spin to select because it's only one
        gui.sel_heisenberg.Enable="off"; //no heisenberg interaction 
        gui.sel_heisenberg2.Enable="off";
        guicheckbox1(maxsl).Value=1;    //the last entry is the Heiseberg coupling
        gui.button_couplingmatrix.Enable="off";
        
        //3rd order plus rate can now be fitted (verstion 1.8)
        gui.button_fit.Enable="on";
        
        gui.entanglement.Visible="off";
        gui.allatomsequ.Visible="off";
        
        if experiment.func_tip then //switch off functionalized tip
            experiment.func_tip=0; //no functionalized tip
            experiment.corr_sample=%f;
            experiment.corr_tip=%f;
            update_gui_values();
            
            ierr = execstr("delete(ts_gui.children)", "errcatch"); 
            //check if figure ts_gui exist
            if ~ierr then
                gui_disable();
                plt_dmatrix();
                gui_enable();                                //yes then update figure
            end
        end
    else                                    //2 or more spin systems
        gui.sel_spin.Enable="on";
        gui.sel_heisenberg.Enable="on";
        gui.sel_heisenberg2.Enable="on";
        guicheckbox1(maxsl).Value=0;    //the last entry is the Heiseberg coupling

//      if temp > 2 | pmodulo(gui.mode1.Value,2)==0 then   //more than 2 atoms can't be fitted
//                                                        //also rates with 2 atoms can't be fitted
        if temp > 2 then   //more than 2 atoms can't be fitted
            gui.button_fit.Enable="off";                   
        else
            gui.button_fit.Enable="on";
        end
        gui.entanglement.Visible="on";
        gui.button_couplingmatrix.Enable="on";
        gui.allatomsequ.Visible="on";
        
        //reshape the interaction matrices for Heisenberg and DM interaction
        //as well as for the Phase factor
        //when the number of spins have changed 
        
        if size(experiment.matrix,1) <> (temp-1) then
            experiment.matrix=resize_matrix(experiment.matrix,temp-1,temp-1);
            experiment.matrix(find(experiment.matrix==''))='0';
        end;
        if size(experiment.matrixDM,1) <> (temp-1) then
            experiment.matrixDM=resize_matrix(experiment.matrixDM,temp-1,temp-1);
            experiment.matrixDM(find(experiment.matrixDM==''))='0, 0, 0';
        end;
        if size(experiment.Phase,1) <> (temp-1) then
            experiment.Phase=resize_matrix(experiment.Phase,temp-1,temp-1);
            experiment.Phase(find(experiment.Phase==''))='0';
        end;

    end;
    
    if  gui.sel_spin.Value > temp then
        gui.sel_spin.value = temp;
        
        if gui_last_spin > temp then
            gui_last_spin = temp;
        end
        
        update_gui_values(); 
        gui.nbr_spins.value = temp;
    end

    if temp<>size(experiment.atom,2) then
        update_atomlink(); 
    end

    if length(experiment.position)==1 then  //tip only interacts with one spin
        if  experiment.position > temp then  
            experiment.position = temp; //avoid having pointer "position" pointing outside its range
        end
    elseif length(experiment.position)>temp then //tip interacts with all spins
        experiment.position=experiment.position(1:temp); //shorten position
        if temp==1 || and(experiment.position==0) then 
            //reset to first spin when only one spin availble or
            //when all values are zero
            experiment.position=1;
        end
        
    elseif length(experiment.position)<temp then //tip interacts with all spins
        experiment.position(length(experiment.position)+1:temp)=0 //expand position
    end
       
    if gui_last_spin > temp then
        gui_last_spin = temp; //avoid having pointer "gui_last_spin" pointing outside its range
    end
    
endfunction

function update_atomlink()
    
    // this function takes the values found in the GUI and updates the 
    // corresponding fields in the global variable "experiment"
    
    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;
    
    global guicheckbox1 guientry1 guitext1 guislider1 gui main_gui experiment ani_gui;

    number_atoms=gui.nbr_spins.value;
    atomlink=experiment.atom;
    size_atomlink=size(atomlink,2);
    
    //check if the number of spin have changed
    if size_atomlink < number_atoms then
        for i=size_atomlink+1:number_atoms,
            atomlink=[atomlink,atomlink(i-1)];   
        end;
        cleardata(2); //all data is now invalid
    elseif size_atomlink > number_atoms then    
        al=atomlink(1);
        for i=2:number_atoms,
            al=[al,atomlink(i)];
        end,
        atomlink=al;
        cleardata(2); //all data is now invalid
    end;
   
    number=gui.sel_spin.Value;
    atomlinkold=atomlink(number);
    
    if gui.allatomsequ.Value==1 & number_atoms >= 2 then
        
        atomlink(:).S=evstr(gui.S.String);
        
        if length(atomlink(number).g)==3 then    //g ist vector
            [a,b]=max(abs(atomlink(number).g));
            temp=atomlink(number).g;
            change=evstr(guientry1(1).String)/temp(b);
            
            if change <> 0 then
                atomlink(:).g=temp*change;
            else
                atomlink(:).g=0;    //reduce to scalar if g=0
                guitext1(1).BackgroundColor = [0.9,0.9,0.9];
                guitext1(1).tooltipstring="g-factor of the spin.";
                guislider1(1).tooltipstring="g-factor of the spin.";
            end
        else
            atomlink(:).g=evstr(guientry1(1).String); //g is scalar
        end

        if atomlink(number).use_Stevens == 0 then
            atomlink(:).D=evstr(guientry1(2).String);
            atomlink(:).E=evstr(guientry1(3).String);
            i=4;
        else
            tempstev=zeros(3,7);
            for i=1:size(%Stevstr,2),
                a=evstr(part(%Stevstr(i),4))/2;
                b=evstr(part(%Stevstr(i),6))+1;
                tempstev(a,b)=evstr(guientry1(1+i).String);
            end
            atomlink(:).Stevens=tempstev
            i=2+size(%Stevstr,2);
        end

        atomlink(:).alpha=evstr(guientry1(i).String);
        atomlink(:).beta=evstr(guientry1(i+1).String);
        atomlink(:).gamma=evstr(guientry1(i+2).String);

        if or(atomlinkold<>atomlink(number)) then
            cleardata(2); //all data is now invalid
            if atomlink(number).use_Stevens <> 0 then
                if (max(atomlink(:).S)>1.5 & max(experiment.atom.S)<2) | ..
                   (max(atomlink(:).S)<2 & max(experiment.atom.S)>1.5) then
                    build_main_gui();
                end
            end
        end
       
        atomlink(:).U=evstr(guientry1(i+3).String);
        
        if length(atomlink(number).J)==3 then    //J ist vector
            [a,b]=max(abs(atomlink(number).J));
            temp=atomlink(number).J;
            change=evstr(guientry1(i+4).String)/temp(b);
            
            if change <> 0 then
                atomlink(:).J=temp*change;
            else
                atomlink(:).J=0;    //reduce to scalar if J=0
                guitext1(i+4).BackgroundColor = [0.9,0.9,0.9];
                guitext1(i+4).tooltipstring="Kondo scattering term.";
                guislider1(i+4).tooltipstring="Kondo scattering term.";
            end
        else
            atomlink(:).J=evstr(guientry1(i+4).String); //j is scalar
        end
    
        atomlink(:).w=evstr(guientry1(i+5).String);

        if or(atomlinkold<>atomlink(number)) then
            cleardata(1); //spectral data is now invalid
            if %renormalize_E then 
                cleardata(2); //all data is invalid
            end,
        end;
    
    else
    
        atomlink(number).S=evstr(gui.S.String);
        
        if length(atomlink(number).g)==3 then    //J ist vector
            [a,b]=max(abs(atomlink(number).g));
            temp=atomlink(number).g;
            change=evstr(guientry1(1).String)/temp(b);
            
            if change <> 0 then
                atomlink(number).g=temp*change;
            else
                atomlink(number).g=0;    //reduce to scalar if zero
                guitext1(1).BackgroundColor = [0.9,0.9,0.9];
                guitext1(1).tooltipstring="Kondo scattering term.";
                guislider1(1).tooltipstring="Kondo scattering term.";
            end
        else
            atomlink(number).g=evstr(guientry1(1).String); //j is scalar
        end

        if atomlink(number).use_Stevens == 0 then
            atomlink(number).D=evstr(guientry1(2).String);
            atomlink(number).E=evstr(guientry1(3).String);
            i=4;
        else
            for i=1:size(%Stevstr,2),
                a=evstr(part(%Stevstr(i),4))/2;
                b=evstr(part(%Stevstr(i),6))+1;
                atomlink(number).Stevens(a,b)=evstr(guientry1(1+i).String);
            end
            i=2+size(%Stevstr,2);
        end

        atomlink(number).alpha=evstr(guientry1(i).String);
        atomlink(number).beta=evstr(guientry1(i+1).String);
        atomlink(number).gamma=evstr(guientry1(i+2).String);
                
        if or(atomlinkold<>atomlink(number)) then
            cleardata(2); //all data is now invalid
        end

        atomlink(number).U=evstr(guientry1(i+3).String);

        if length(atomlink(number).J)==3 then    //J ist vector
            [a,b]=max(abs(atomlink(number).J));
            temp=atomlink(number).J;
            change=evstr(guientry1(i+4).String)/temp(b);
            
            if change <> 0 then
                atomlink(number).J=temp*change;
            else
                atomlink(number).J=0;    //reduce to scalar if zero
                guitext1(i+4).BackgroundColor = [0.9,0.9,0.9];
                guitext1(i+4).tooltipstring="Kondo scattering term.";
                guislider1(i+4).tooltipstring="Kondo scattering term.";
            end
        else
            atomlink(number).J=evstr(guientry1(i+4).String); //j is scalar
        end

        atomlink(number).w=evstr(guientry1(i+5).String);
        
        if or(atomlinkold<>atomlink(number)) then
            cleardata(1); //spectral data is now invalid
            if %renormalize_E then 
                cleardata(2); //all data is invalid
            end,
        end
        
    end
  
    experiment.atom=atomlink;    //play it back to experiment
                                          
    //check if figure ani_gui exist
    ierr = execstr("delete(ani_gui.children)", "errcatch"); 
    if ~ierr then
       gui_disable();
       gui_checkanisotropy(); //yes then update figure
       gui_enable();
    end

    old=experiment.heisenberg_coupling;
    experiment.heisenberg_coupling=evstr(gui.sel_heisenberg.string);    

    if old<>experiment.heisenberg_coupling then
        cleardata(2); //all data is now invalid
    end
    
    old=experiment.B;
    experiment.B=..
    [evstr(guientry1(i+14).String),evstr(guientry1(i+13).String),evstr(guientry1(i+12).String)];
    if or(old<>experiment.B) then
        cleardata(2); //all data is now invalid
        
        //check if figure ani_gui exist
        ierr = execstr("delete(ani_gui.children)", "errcatch"); 
        if ~ierr then
           gui_disable();
           gui_checkanisotropy(); //yes then update figure
           gui_enable();
        end
    end
    
    old=experiment.ptip;    
    if experiment.paramagnetic then
        //calculate the splitting energy
        delta=sqrt(sum(experiment.B.^2))*%muBohr*experiment.paramag_g*experiment.paramag_S;
        if delta<>0 then
            S.S=experiment.paramag_S;
            polarization=AvrgM(S,delta,experiment.T)*experiment.eta;
            experiment.ptip=polarization*experiment.B/sqrt(sum(experiment.B.^2));
            // update gui
            guientry1(i+11).String=string(polarization);
            update_slider();
        else
            experiment.ptip=[0 0 0];
            // update gui
            guientry1(i+11).String='0';
            guislider1(i+11).Value=50;
        end;
    else
        if sum(experiment.B.^2)<>0 then
        
            if sum(experiment.ptip.^2)<>0 then
                //calculate angle between B and ptip
                angle=real(acos((experiment.ptip*experiment.B')..
                /(sqrt(sum(experiment.ptip.^2))*sqrt(sum(experiment.B.^2)))));
                sig=1-2*(angle>%pi/2); //if angle>pi/2 then sig is -1, else +1;
            
                experiment.ptip=experiment.ptip/sqrt(sum(experiment.ptip.^2))..
                *evstr(guientry1(i+11).string)*sig;
        
            else
                //allign in direction of B
                experiment.ptip=experiment.B/sqrt(sum(experiment.B.^2))..
                *evstr(guientry1(i+11).string);
            end
        else
            experiment.ptip=[0 0 0];
        end
    end
    if or(old<>experiment.ptip) then
        cleardata(1);
        //check angle between B and p_tip
        angle=real(acos((experiment.ptip*experiment.B')..
        /(sqrt(sum(experiment.ptip.^2))*sqrt(sum(experiment.B.^2)))));
        if isnan(angle) then
            angle=0;
        end
        if clean(angle)<>0 & abs(clean(angle))<>%pi then
            guitext1(i+11).backgroundcolor=[1,0.5,0.5];
            guitext1(i+11).tooltipstring="Spin polarization in the tip. ATTENTION not alligned with B!";
            guislider1(i+11).tooltipstring="Spin polarization in the tip. ATTENTION not alligned with B!";
        else
            guitext1(i+11).backgroundcolor=[0.9,0.9,0.9];
            guitext1(i+11).tooltipstring="Spin polarization in the tip";
            guislider1(i+11).tooltipstring="Spin polarization in the tip";
        end
                
        ierr = execstr("delete(ts_gui.children)", "errcatch"); 
        //check if figure ts_gui exist
        if ~ierr then
           gui_disable();
           plt_dmatrix();
           gui_enable();                                //yes then update figure
        end
    end
    
    if experiment.use_T_ts then
        old=experiment.T_ts;
        [a,b]=max(abs(experiment.T_ts));
        temp=experiment.T_ts;
        change=evstr(guientry1(i+6).String)/temp(b); 
        experiment.T_ts=temp*change;

        if experiment.T_ts <> old then            
                cleardata(1); //spectral data is now invalid 
        end            
    else        
        old=experiment.A;
        experiment.A=evstr(guientry1(i+6).String);
        if old<>experiment.A then
            if experiment.rate_calc then 
                cleardata(1); //spectral data is now invalid 
            else
                cleardata(0); //only delete spec.calc
            end;
        end
    end

    old=experiment.b;
    experiment.b=evstr(guientry1(i+7).String);
    if old<>experiment.b then 
        cleardata(0); //only delete spec.calc
    end
    
    old=experiment.x0;
    experiment.x0=evstr(guientry1(i+8).String);
    if old<>experiment.x0 then 
        cleardata(0); //only delete spec.calc
    end
    
    old=experiment.y0;
    experiment.y0=evstr(guientry1(i+9).String);
    if old<>experiment.y0 then 
        cleardata(0); //only delete spec.calc
    end
    
    old=experiment.xrange;
    experiment.xrange=evstr(guientry1(i+15).String);
    if experiment.xrange<=0 then 
        experiment.xrange=old;  //xrange >0!
        guientry1(i+15).String=string(old);
    end
    if old<>experiment.xrange then 
        cleardata(1); //spectral data is now invalid 
    end
    
    old=experiment.T;
    experiment.T=%kBoltzmann*evstr(guientry1(i+10).String);
    if old<>experiment.T then 
        cleardata(1); //spectral data is now invalid 
    end
    
    if atomlink(number).use_Stevens==0 then
        //check that 3*|E|<=|D|. If not make the star "red"
        if 3*abs(atomlink(number).E)>abs(atomlink(number).D) | atomlink(number).E<0 then
            gui.button_anisotropy.foregroundcolor=[1 0.2 0.2]
        else
            gui.button_anisotropy.foregroundcolor=[0 0 0];
        end
    end
    
    if atomlinkold.S<>atomlink(number).S & atomlink(number).use_Stevens <> 0 then
        build_main_gui();
    end

        
endfunction
    
function update_gui_values()
    
    //function to renew and update the values shown in the GUI using the
    //values in the global variable 'experiment'
    
    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;
    
    global guicheckbox1 guientry1 guitext1 guislider1 gui main_gui experiment;
    
    gui_disable();  //disable manipulation of the GUI
    
    atomlink=experiment.atom;
    gui.nbr_spins.value=size(atomlink,2);
    gui.sel_heisenberg.String=string(experiment.heisenberg_coupling);
    
    gui.entanglement.value=1*(experiment.entanglement);
    gui.mode1.value=4*(experiment.third_order_calc)+1*(experiment.rate_calc)+1;
    gui.spectrum_deriv.value=1*experiment.spectrum_deriv+1;
    gui.allatomsequ.value=1*(experiment.allatomsequ);
    
    number=gui.sel_spin.Value;
    if number>size(atomlink,2) then
        number=size(atomlink,2);
    end
    
    gui.S.String=string(atomlink(number).S);
    
    if length(atomlink(number).g)==3 then    //g ist vector
        [a,b]=max(abs(atomlink(number).g));
        temp=atomlink(number).g;
        guientry1(1).String=string(temp(b));  //shows the largest absolute value
        guitext1(1).BackgroundColor = [1,0.5,0.5]; 
        guitext1(1).tooltipstring="g-factor of the spin. ATTENTION anisotropic factor! Only principal value shown!";
        guislider1(1).tooltipstring="g-factor of the spin. ATTENTION anisotropic factor! Only principal value shown!";
    else
        guientry1(1).String=string(atomlink(number).g);        
        guitext1(1).BackgroundColor = [0.9,0.9,0.9];
        guitext1(1).tooltipstring="g-factor of the spin.";
        guislider1(1).tooltipstring="g-factor of the spin.";
    end
    
    if atomlink(number).use_Stevens==0 then
        guientry1(2).String=string(atomlink(number).D);
        guientry1(3).String=string(atomlink(number).E);
        i=4
    else
        for i=1:size(%Stevstr,2),
            a=evstr(part(%Stevstr(i),4))/2;
            b=evstr(part(%Stevstr(i),6))+1;
            guientry1(1+i).String=string(atomlink(number).Stevens(a,b));
        end
        i=2+size(%Stevstr,2);
    end
    
    guientry1(i).String=string(atomlink(number).alpha);
    guientry1(i+1).String=string(atomlink(number).beta);
    guientry1(i+2).String=string(atomlink(number).gamma);
    guientry1(i+3).String=string(atomlink(number).U);
    
    if length(atomlink(number).J)==3 then    //J ist vector
        [a,b]=max(abs(atomlink(number).J));
        temp=atomlink(number).J;
        guientry1(i+4).String=string(temp(b));  //shows the largest absolute value
        guitext1(i+4).BackgroundColor = [1,0.5,0.5]; 
        guitext1(i+4).tooltipstring="Kondo scattering term. ATTENTION anisotropic coupling! Only principal value shown!";
        guislider1(i+4).tooltipstring="Kondo scattering term. ATTENTION anisotropic coupling! Only principal value shown!";
    else
        guientry1(i+4).String=string(atomlink(number).J);        
        guitext1(i+4).BackgroundColor = [0.9,0.9,0.9];
        guitext1(i+4).tooltipstring="Kondo scattering term.";
        guislider1(i+4).tooltipstring="Kondo scattering term.";
    end
    
    guientry1(i+5).String=string(atomlink(number).w);
    
    if experiment.use_T_ts then
        temp=experiment.T_ts;
        [a,b]=max(abs(temp)); //find the absolute largest value
        if temp(1)==temp(2) & temp(3)==temp(2) then //all entries are equal
            guientry1(i+6).String=string(temp(1));        
            guitext1(i+6).BackgroundColor = [0.9,0.9,0.9];
            guitext1(i+6).tooltipstring="Exchange coupling between tip and sample.";
            guislider1(i+6).tooltipstring="Exchange coupling between tip and sample.";
        else
            guientry1(i+6).String=string(temp(b));  //shows the largest absolute value
            guitext1(i+6).BackgroundColor = [1,0.5,0.5]; 
            guitext1(i+6).tooltipstring="Exchange coupling between tip and sample. ATTENTION anisotropic coupling! Only principal value shown!";
            guislider1(i+6).tooltipstring="Exchange coupling between tip and sample. ATTENTION anisotropic coupling! Only principal value shown!";
        end
    else
        guientry1(i+6).String=string(experiment.A);        
    end

    guientry1(i+7).String=string(experiment.b);
    guientry1(i+8).String=string(experiment.x0);
    guientry1(i+9).String=string(experiment.y0);
    guientry1(i+10).String=string(experiment.T/%kBoltzmann);
    guientry1(i+15).String=string(max(experiment.xrange));
    
    //calculate angle between B and ptip
    angle=real(acos((experiment.ptip*experiment.B')..
    /(sqrt(sum(experiment.ptip.^2))*sqrt(sum(experiment.B.^2)))));
    //if angle is larger than 90 deg eta<0
    //eta is the length of experiment.ptip
    if isnan(angle) then 
        angle=0;
    end
    if angle>%pi/2 then
        guientry1(i+11).String=string(-sqrt(sum(experiment.ptip.^2)));
    else
        guientry1(i+11).String=string(sqrt(sum(experiment.ptip.^2)));
    end
    
    if clean(angle)<>0 & abs(clean(angle))<>%pi then 
        guitext1(i+11).BackgroundColor = [1,0.5,0.5]; 
        guitext1(i+11).tooltipstring="Spin polarization in the tip. ATTENTION not alligned with B!";
        guislider1(i+11).tooltipstring="Spin polarization in the tip. ATTENTION not alligned with B!";
    else
        guitext1(i+11).BackgroundColor = [0.9,0.9,0.9];
        guitext1(i+11).tooltipstring="Spin polarization in the tip.";
        guislider1(i+11).tooltipstring="Spin polarization in the tip.";
    end
    
    guientry1(i+14).String=string(experiment.B(1));
    guientry1(i+13).String=string(experiment.B(2));
    guientry1(i+12).String=string(experiment.B(3));
    
    //change color when parameters of the spin on the tip are diplayed
    if experiment.func_tip & gui.sel_spin.value==1 then
        guientry1(1:i+5).backgroundcolor=[1 0.8 0.8];
        gui.S.backgroundcolor=[1 0.8 0.8];
        gui.sel_spin.backgroundcolor=[1,0.8,0.8];
        if experiment.func_tip==1 then
            gui.spin_nr.string="Tip Spin (dark)";
            else
            gui.spin_nr.string="Tip Spin (bright)";
        end
    
        else
        guientry1(1:i+5).backgroundcolor=[1 1 1];
        gui.S.backgroundcolor=[1 1 1];
        gui.sel_spin.backgroundcolor=[1,1,1];
        gui.spin_nr.string="Sample Spin #";
    end
   
    if atomlink(number).use_Stevens==0 then
        //check that 3*|E|<=|D|. If not make the star "red"
        if 3*abs(atomlink(number).E)>abs(atomlink(number).D) | atomlink(number).E<0 then
            gui.button_anisotropy.foregroundcolor=[1 0.2 0.2]
        else
            gui.button_anisotropy.foregroundcolor=[0 0 0];
        end
    end

    update_slider();
    gui_setmode();
    
    gui_enable();
    
endfunction

function gui_setmode()
    
    //function triggered when certain modes are changed:
    //
    // - the "entanglement" checkbox which enable substrate scattering also with
    //      other spin system (important for 3rd order calc, rate equation, and 
    //      BlochRedfield calculations)
    // - the "Order" pulldown menue (2nd , 3rd , 2nd+rate, 3rd+rate)
    // - the "derivative of spectra" pulldown menue (I(V), dI/dV(V), dI2/d2V(V))
    // - the "all atoms are equal" checkbox
    //
    // the routine will change the global variable "experiment" and delete if 
    // necessary the Data via the routine "cleardata()"
    
    if exists("%guidebug") then 
        [linenum, callername] = where();
        disp(callername(1));
    end;

    global gui experiment;
    
    oldval=experiment.entanglement;
    if gui.entanglement.Value==1 then
        experiment.entanglement=%T;
    else
        experiment.entanglement=%F;
    end;
    if oldval<>experiment.entanglement then
        cleardata(1); //spectral data is now invalid 
    end

    oldval=experiment.spectrum_deriv;
    experiment.spectrum_deriv=gui.spectrum_deriv.value-1;
    if oldval<>experiment.spectrum_deriv then
        cleardata(1); //spectral data is now invalid 
        if experiment.spectrum_deriv == 0 & 1*experiment.rate_calc==0 then   
            experiment.rate.calc=%T; //we need rate equations when calculating I(V)
            gui.mode1.Value=gui.mode1.Value+1;
        end
        
    end
    
    oldval=experiment.rate_calc;
    experiment.rate_calc=pmodulo(gui.mode1.Value-1,4) 
    
    if experiment.rate_calc==0 then
        if experiment.spectrum_deriv == 0 then
            experiment.spectrum_deriv =1;
            gui.spectrum_deriv.value=2;
        end;
    end;
    
    if gui.mode1.Value==4 then  //Bloch-Redfield calculation
       gui.button_bloch_redfield.Enable="on";
       if pmodulo(gui.mode1.Value-1,4)<>oldval then
           //switch on entanglement as default
           experiment.entanglement=%T;
           gui.entanglement.Value=1;
           cleardata(1);
       end
    else
       gui.button_bloch_redfield.Enable="off";
    end

    if oldval<>experiment.rate_calc then
        cleardata(1); //spectral data is now invalid 
    end
    
    oldval=experiment.third_order_calc;
    if gui.mode1.Value>4 then
        experiment.third_order_calc=%T;
    else
        experiment.third_order_calc=%F;
    end;
    if oldval<>experiment.third_order_calc then
        cleardata(1); //spectral data is now invalid 
    end
    
    oldval=experiment.allatomsequ;
    if gui.allatomsequ.value==1 then
        experiment.allatomsequ=%T;
        if oldval<>experiment.allatomsequ & size(experiment.atom,2)>1 then
            cleardata(2); //all data is now invalid 
            update_atomlink();
        end
    else
        experiment.allatomsequ=%F;
    end;
    
    if gui.nbr_spins.value>2 then
       gui.button_fit.Enable="off";
    else
       gui.button_fit.Enable="on";
    end;

endfunction

function gui_tipsample()
    
   //opens the tip-sample submenue

   if exists("%guidebug") then 
       [linenum, callername] = where();
       disp(callername(1));
   end;
    
   global experiment gui gg ts_gui;
   
   ierr = execstr("ts_gui.figure_id", "errcatch");
   if ierr<>0 then
       ts_gui=figure('figure_size',[500,670],'auto_resize',..
       'on','figure_name',..
       '3rd order tunneling simulator. (c) Markus Ternes ',..
       'resize','off');
        plt_dmatrix();
   else
       show_window(ts_gui);
   end
    
endfunction
       
function plt_dmatrix()
    
   //function drawing the tip-sample submenue
    
   global experiment guicheckbox1 guientry1 guislider1 gui ts_gui gg;
   
   drawlater();
   scf(ts_gui);
   subplot(121);
   mat=dmatrix(experiment.ptip);
   mat=round(mat*1000)/1000;
   strmat=strsubst(string(mat), '%i*', 'i\times ');
   str="$\\ \mbox{Density matrix:}\\ \begin{pmatrix}"+strmat(1,1)+..
   "&"+strmat(2,1)+"\\"..
   +strmat(2,1)+"&"+strmat(2,2)+"\end{pmatrix}$";
   xtitle(str);
   blochsphere(mat,'red');
   subplot(122);
   mat=dmatrix(experiment.psample);
   mat=round(mat*1000)/1000;
   strmat=strsubst(string(mat), '%i*', 'i\times ');
   str="$\\ \mbox{Density matrix:}\\ \begin{pmatrix}"+strmat(1,1)+..
   "&"+strmat(2,1)+"\\"..
   +strmat(2,1)+"&"+strmat(2,2)+"\end{pmatrix}$";
   xtitle(str);
   blochsphere(mat,'blue');
   drawnow();
    
   uicontrol("parent",ts_gui, "relief","groove", ..
   "style","frame", "units","pixels", ..
   "position",[10, 5, 480, 110], ..
   "horizontalalignment","center", "background",[0.9 0.9 0.9], ..
   "tag","frame_control");
   
   gg.data=[];

   gg.button_change = uicontrol("parent",ts_gui, "style","pushbutton", ..
   "Position",[200-20 45 100 25], "String","Polarization", ..
   "BackgroundColor",[0.6,0.9,0.9], "fontsize",12, ..
   "Relief","raised",..
   "TooltipString", "Change spin polarization in tip and sample", ..
   "Callback","gui_ts_change();");
   
   gg.button_change2 = uicontrol("parent",ts_gui, "style","pushbutton", ..
   "Position",[200-20 80 45 25], "String","J_s", ..
   "BackgroundColor",[0.6,0.9,0.9], "fontsize",12, ..
   "Relief","raised",..
   "TooltipString", "Change Kondo coupling between atom and sample", ..
   "Callback","gui_Js_change();");
   
   if experiment.use_T_ts then
       bc=[0.6,0.9,0.9]; 
   else
       bc=[0.9,0.9,0.9]; 
   end
   
   gg.button_change3 = uicontrol("parent",ts_gui, "style","pushbutton", ..
   "Position",[255-20 80 45 25], "String","T_ts", ..
   "BackgroundColor",bc, "fontsize",12, ..
   "Relief","raised",..
   "TooltipString", "Change exchange coupling between tip and sample",..
   "Callback","gui_T_ts_change();");

   gg.button_ok = uicontrol("parent",ts_gui, "style","pushbutton", ..
   "Position",[200-20 10 100 25], "String","OK", ..
   "BackgroundColor",[0.6,0.9,0.6], "fontsize",12, ..
   "Relief","raised",..
   "TooltipString", "Close submenu",..
   "Callback","close(ts_gui)");

   uicontrol("parent",ts_gui, "style","text",..
   "string","Tip", "position",[60-20,80,140,30], ..
   "horizontalalignment","center", "fontsize",18, ..
   "background",[0.9 0.9 0.9]);

   uicontrol("parent",ts_gui, "style","text",..
   "string","Sample", "position",[300,80,140,30], ..
   "horizontalalignment","center", "fontsize",18, ..
   "background",[0.9 0.9 0.9]);

   uicontrol("parent",ts_gui, "style","checkbox",..
   "string","Paramag.", "position",[80,10,100,25], ..
   "horizontalalignment","center", "fontsize",12,'Value',..
   1*experiment.paramagnetic,"background",[0.9 0.9 0.9],..
    "TooltipString", "Paramagnetic tip. Polarization will be calculated",..
    "Callback","experiment.paramagnetic=~experiment.paramagnetic;ts_para_callback();");

   gg.func_tip = uicontrol("parent",ts_gui, "Position", [310-20,10,100+20,25],..
   "Style", "popupmenu",..
   "TooltipString", "Select functionalized tip",..
   "BackgroundColor",[0.6,0.9,0.9], "fontsize",11, ..
   "Relief","raised",..
   "String", "Reg. tip|Fnc. tip (dark)|Fnc. tip (bright)",..
   "Callback", "cleardata(1);ts_para_callback()",..
   "value", experiment.func_tip+1);

//    uicontrol("parent",ts_gui, "style","checkbox",..
//    "string","FT", "position",[300,10,60,25], ..
//    "horizontalalignment","center", "fontsize",12,'Value',..
//    1*experiment.func_tip,"background",[0.9 0.9 0.9],..
//    "TooltipString", "Functionalized tip",..
//    "Callback",..
//    "experiment.func_tip=~experiment.func_tip;cleardata(1);ts_para_callback()");
    
    uicontrol("parent",ts_gui, "style","checkbox",..
    "string","CS", "position",[420,40,60,25], ..
    "horizontalalignment","left", "fontsize",12,'Value',..
    1*experiment.corr_sample,"background",[0.9 0.9 0.9],..
    "TooltipString", "Correlation in Sample",..
    "Callback",..
    "experiment.corr_sample=~experiment.corr_sample;cleardata(1);ts_para_callback()");
    
    uicontrol("parent",ts_gui, "style","checkbox",..
    "string","CT", "position",[420,10,60,25], ..
    "horizontalalignment","left", "fontsize",12,'Value',..
    1*experiment.corr_tip,"background",[0.9 0.9 0.9],..
    "TooltipString", "Correlation in Tip",..
    "Callback",..
    "experiment.corr_tip=~experiment.corr_tip;cleardata(1);ts_para_callback()");

    uicontrol("parent",ts_gui, "style","text",..
    "string","$g$", "position",[25,40,140,25], ..
    "horizontalalignment","left", "fontsize",12, ..
    "TooltipString", "g-factor of the tip",..
    "background",[0.9 0.9 0.9]);

    gg.gui_paramag_g=uicontrol("parent",ts_gui, "style","edit",..
    "string",string(experiment.paramag_g), "position",[40,45,40,20], ..
    "horizontalalignment","center", "fontsize",12, ..
    "background",[1 1 1], "tag","para_g",..
    "Enable",1*experiment.paramagnetic,..
    "TooltipString", "g-factor of the tip",..
    "Callback",'ts_para_callback()');

    uicontrol("parent",ts_gui, "style","text",..
    "string","$S$", "position",[25,10,40,25], ..
    "horizontalalignment","center", "fontsize",12, ..
    "TooltipString", "Total spin of tip",..
    "background",[0.9 0.9 0.9]);

    gg.gui_paramag_S=uicontrol("parent",ts_gui, "style","edit",..
    "string",string(experiment.paramag_S), "position",[40,15,40,20], ..
    "horizontalalignment","center", "fontsize",12, ..
    "background",[1 1 1], "tag","para_g",..
    "Enable",1*experiment.paramagnetic,..
    "TooltipString", "Total spin of tip",..
    "Callback",'ts_para_callback()');

    uicontrol("parent",ts_gui, "style","text",..
    "string","$\eta_{max}$", "position",[105-20,40,40,25], ..
    "horizontalalignment","left", "fontsize",12, ..
    "TooltipString", "Maximal spin polarization of tip",..
    "background",[0.9 0.9 0.9]);
    
    gg.gui_eta=uicontrol("parent",ts_gui,"style","edit",..
    "string",string(experiment.eta), "position",[140-20,45,40,20], ..
    "horizontalalignment","center", "fontsize",12, ..
    "background",[1 1 1], "tag","para_eta",..
    "Enable",1*experiment.paramagnetic,..
    "TooltipString", "Maximal spin polarization of tip",..
    "Callback",'ts_para_callback()');

    uicontrol("parent",ts_gui, "style","text",..
    "string","$J\rho_0\times$", "position",[315-20,40,40,25], ..
    "TooltipString", "Scaling factor",..
    "horizontalalignment","left", "fontsize",12, ..
    "background",[0.9 0.9 0.9]);

    gg.gui_sef=uicontrol("parent",ts_gui,"style","edit",..
    "string",string(experiment.sef), "position",[355-20,45,40+20,20], ..
    "horizontalalignment","center", "fontsize",12, ..
    "background",[1 1 1], "tag","sef","Callback",..
    'ts_para_callback()');
    
endfunction

function gui_ts_change()
    
   //submenue to change spin polarization of tip and sample 
   
   global experiment guicheckbox1 guientry1 guislider1 gui ts_gui;

   labelh=["Tip","Sample"];
   labelv=["x","y","z"];
   str=string([experiment.ptip;experiment.psample])';
   new=x_mdialog("Spin Polarisation",..
   labelv,labelh,str);
   if and(size(new)==[3,2]) then
       new(~isnum(new))='0'; //replace bullshit and set it to '0'  
       if or(new<>str) then
          experiment.ptip=evstr(new(:,1))';
          experiment.psample=evstr(new(:,2))';
          cleardata(1);
       end
   end
   amp=sqrt(sum(experiment.ptip.^2));
   if amp>1 then
       experiment.ptip=experiment.ptip/amp;
   end
   amp=sqrt(sum(experiment.psample.^2));
   if amp>1 then
       experiment.psample=experiment.psample/amp;
   end
   ierr = execstr("delete(ts_gui.children)", "errcatch"); 
       //check if figure ts_gui exist
   if ~ierr then
      gui_disable();
      plt_dmatrix();
      gui_enable();                                //yes then update figure
      show_window(ts_gui.figure_id)                //make it visible
   else
      gui_tipsample();                              //no then open figure
   end

   ts_para_callback();
   
endfunction

function gui_g_change()
    
   //submenue to change the g-factor between localized spin 
   //and sample electron bath   
   
   global experiment guicheckbox1 guientry1 guislider1 gui ts_gui;

   number=gui.sel_spin.Value;
   allatomsequ=(experiment.allatomsequ & size(experiment.atom,2)>1);
   
   if allatomsequ then
       labelh=["g-factor for all spins"];
   else
      labelh=["g-factor for spin # "+string(number)];
   end
 
   labelv=["g_x","g_y","g_z"];
   str=string(experiment.atom(number).g)';
   if size(str,1)==1 then
       str=[str; str; str];
   end
   new=x_mdialog("g-factor",..
   labelv,labelh,str);
   if and(size(new)==[3,1]) then
       new(~isnum(new))='0'; //replace bullshit and set it to '0'  
       if or(new<>str) then
          if new(1)==new(2) & new(2)==new(3) then //all entries are the same; reduce to scalar
              if allatomsequ then
                  experiment.atom(:).g=evstr(new(1));
              else
                  experiment.atom(number).g=evstr(new(1));
              end
          else
              if allatomsequ then
                  experiment.atom(:).g=evstr(new(:))';
              else
                  experiment.atom(number).g=evstr(new(:))';
              end
          end
          cleardata(1);
          update_gui_values;
       end
   end
  
endfunction

function gui_Js_change()
    
   //submenue to change Kondo coupling between localized spin 
   //and sample electron bath   
   
   global experiment guicheckbox1 guientry1 guislider1 gui ts_gui;

   number=gui.sel_spin.Value;
   allatomsequ=(experiment.allatomsequ & size(experiment.atom,2)>1);
   
   if allatomsequ then
       labelh=["Kondo coupling to sample for all spins"];
   else
      labelh=["Kondo coupling to sample for spin # "+string(number)];
   end
 
   labelv=["J_x*rho_s","J_y*rho_s","J_z*rho_s"];
   str=string(experiment.atom(number).J)';
   if size(str,1)==1 then
       str=[str; str; str];
   end
   new=x_mdialog("Kondo coupling",..
   labelv,labelh,str);
   if and(size(new)==[3,1]) then
       new(~isnum(new))='0'; //replace bullshit and set it to '0'  
       if or(new<>str) then
          if new(1)==new(2) & new(2)==new(3) then //all entries are the same; reduce to scalar
              if allatomsequ then
                  experiment.atom(:).J=evstr(new(1));
              else
                  experiment.atom(number).J=evstr(new(1));
              end
          else
              if allatomsequ then
                  experiment.atom(:).J=evstr(new(:))';
              else
                  experiment.atom(number).J=evstr(new(:))';
              end
          end
          cleardata(1);
          update_gui_values;
       end
   end
  
endfunction

function gui_T_ts_change()
    
   //submenue for changing exchange coupling between tip and sample electrode
   
   global experiment guicheckbox1 guientry1 guislider1 gui ts_gui;

   if  experiment.use_T_ts==%f then
       messagebox("Switch on ''use_T_ts'' under ''Parameter'' to use this function");
   else
       labelv=["T_ts_x","T_ts_y","T_ts_z"];
       labelh=["Exchange coupling between tip and sample"]
       str=string(experiment.T_ts)';
       new=x_mdialog("Exchange coupling between tip and sample",..
       labelv,labelh,str);
       if and(size(new)==[3,1]) then
          new(~isnum(new))='0'; //replace bullshit and set it to '0'  
          if or(new<>str) then
              experiment.T_ts=evstr(new)';
              cleardata(1);
              update_gui_values;
          end
       end
   end

endfunction


function ts_para_callback()
    
    global experiment gg ts_gui guicheckbox1 guientry1 guislider1 gui;

    str=gg.gui_eta.String;
    str(~isnum(str))='0'; //replace bullshit and set it to '0'
    if experiment.eta<>evstr(str) then
        experiment.eta=evstr(str);
        cleardata(1);
    end
    str=gg.gui_paramag_g.String;
    str(~isnum(str))='2'; //replace bullshit and set it to '2'
    if experiment.paramag_g<>evstr(str) then
        experiment.paramag_g=evstr(str);
        cleardata(1);
    end
    
    str=gg.gui_paramag_S.String;
    str(~isnum(str))='0.5'; //replace bullshit and set it to '0.5'
    if experiment.paramag_S<>evstr(str) then
        experiment.paramag_S=evstr(str);
        cleardata(1);
    end

    str=gg.gui_sef.String;
    str(~isnum(str))='1'; //replace bullshit and set it to '1'
    if experiment.sef<>evstr(str) then
        experiment.sef=evstr(str);
        cleardata(1);
    end
    
    experiment.func_tip=gg.func_tip.value-1;
    
    if experiment.func_tip & size(experiment.atom,2)==1  then
        gui.nbr_spins.value=2;   
        //increase number of spins to at least 2 when we have a "func-tip"
        number_spins_callback(); 
    end
    
    gg.gui_paramag_g.enable=1*experiment.paramagnetic;
    gg.gui_paramag_S.enable=1*experiment.paramagnetic;
    gg.gui_eta.enable=1*experiment.paramagnetic;

    update_gui_values();
    update_atomlink();
   
endfunction

function gui_couplingmatrix()
    
   global experiment ok cm_gui hh;

   nbr=size(experiment.atom,2); //number of spins
   
   ierr = execstr("cm_gui.figure_id", "errcatch");
   
   if ierr<>0 then
  
       cm_gui=figure('figure_size',[530,530],'auto_resize',..
       'on','figure_name',..
       '3rd order tunneling simulator. (c) Markus Ternes ');
       scf(cm_gui);
       
    //   hh.data=[];
       hh.graph_ui=newaxes();
       hh.graph_ui.margins = [ 0 0 0 0];
       hh.graph_ui.axes_bounds = [0.1,0.1,0.8,0.8];
       plot2d(0,0,-1,"010"," ",[-1.5,-1.5,1.5,1.5]);
       H_sw=0;
       
       hh.button_change1 = uicontrol("parent",cm_gui, "style","pushbutton", ..
       "Position",[300 10 70 25], "String","Heisenberg", ..
       "BackgroundColor",[0.6,0.9,0.9], "fontsize",11, ..
       "Relief","raised",..
       "TooltipString",..
       "Change Heisenberg coupling constants Use one value of isotropic, otherwise three values",..
       "Callback","change_coupling(1)");
           
       hh.button_change2 = uicontrol("parent",cm_gui, "style","pushbutton", ..
       "Position",[220 10 70 25], "String","DM", ..
       "BackgroundColor",[0.6,0.9,0.9], "fontsize",11, ..
       "Relief","raised",..
       "TooltipString", "Change antisymmetric (Dzyaloshinskii-Moriya) exchange constants",..
       "Callback","change_coupling(2)");
       
       hh.button_change3 = uicontrol("parent",cm_gui, "style","pushbutton", ..
       "Position",[140 10 70 25], "String","Phase", ..
       "BackgroundColor",[0.9,0.9,0.6], "fontsize",11, ..
       "Relief","raised",..
       "TooltipString", "Change the scattering phase between sites",..
       "Callback","change_coupling(3)");
       
       hh.button_change4 = uicontrol("parent",cm_gui, "style","pushbutton", ..
       "Position",[55 10 75 25], "String","Tip Position", ..
       "BackgroundColor",[0.9,0.6,0.6], "fontsize",11, ..
       "Relief","raised",..
       "TooltipString", "Change the tip position",..
       "Callback","change_tip_position");

    
       hh.button_ok = uicontrol("parent",cm_gui, "style","pushbutton", ..
       "Position",[390 10 80 25], "String","OK", ..
       "BackgroundColor",[0.6,0.9,0.6], "fontsize",11, ..
       "Relief","raised",..
       "TooltipString", "Close submenu",..
       "Callback","close(cm_gui)");
   
   else
       show_window(cm_gui.figure_id);
   end
   
    plt_coupling();

endfunction

function plt_coupling()
   
   global experiment cm_gui hh;

   mH=experiment.matrix;
   mDM=experiment.matrixDM;
   atomlink=experiment.atom;
   nbr=size(atomlink,2); //number of spins
   scf(cm_gui);
   if hh.graph_ui.children<>[] then
       delete(hh.graph_ui.children);
   end;
   xv=[];yv=[];z=[];
   for i=1:size(mH,1),
      for j=1:size(mH,2),
         xvv=[sin(2*%pi/nbr*(i-1));sin(2*%pi/nbr*j)];
         yvv=[cos(2*%pi/nbr*(i-1));cos(2*%pi/nbr*j)];
         if sum(evstr(mH(i,j)).^2)<>0 then,
            xv=[xv,xvv];
            yv=[yv,yvv];
            if real(sum(evstr(mH(i,j))))>0 then,
               z=[z,1]; //positive entrace: black line (color=1) 
            else
               z=[z,5]; //negative entrance: red line (color=5)
            end;
             
         end;
         if sum(evstr(mDM(i,j)).^2)<>0 then,
             x=[];y=[];
             for k=0:0.01:1,
                dy=yvv(1)-yvv(2);
                dx=xvv(1)-xvv(2);
                l=sqrt(dx^2+dy^2);
                x=[x;xvv(2)+k*dx+0.03*sin(k*8*%pi*l)*dy/l];
                y=[y;yvv(2)+k*dy-0.03*sin(k*8*%pi*l)*dx/l];  
             end;
             plot2d(x,y,2,'000');
         end;
      end;
   end;
   
   xsegs(xv,yv,abs(z));
   
   //plot filled, color-coded circles for the spins  
   for i=0:nbr-1,
//     xset("color",atomlink.S(i+1)*2):
     a=gca();
     a.foreground = atomlink.S(i+1)*2;

     if a.foreground==8 then
         a.foreground=a.foreground+1;   //avoid white as color
     end
     
     xfarc(sin(2*%pi/nbr*i)-0.1,cos(2*%pi/nbr*i)+0.1,0.2,0.2,0,360*64);
     xstring(1.25*sin(2*%pi/nbr*i)-0.15,1.25*cos(2*%pi/nbr*i)-0.1,"$#\!:\!"+string(i+1)+"$") 
   end

   pos=experiment.position;
   if length(pos)==1 then   //only one spin couples to the tip
       a=gca();
       a.foreground=1;
       xarc(sin(2*%pi/nbr*(pos-1))-0.15,cos(2*%pi/nbr*(pos-1))+0.15,0.3,0.3,0,360*64);
   elseif length(pos)==nbr then
        blckcircle=(abs(pos)==max(abs(pos)));
        grycircle=(pos<>0&~blckcircle);
        for i=0:nbr-1,
            if blckcircle(i+1) then
               a=gca();
               a.foreground=1;
               xarc(sin(2*%pi/nbr*i)-0.15,cos(2*%pi/nbr*i)+0.15,0.3,0.3,0,360*64);
            elseif grycircle(i+1) then
               a=gca();
               a.foreground=33;
               xarc(sin(2*%pi/nbr*i)-0.15,cos(2*%pi/nbr*i)+0.15,0.3,0.3,0,360*64);
            end
        end
   end
     
endfunction

function change_tip_position()
    
    global experiment cm_gui hh;
    c=%F
    while c==%F
       tip_pos=evstr(x_dialog('Tip position(s)',string(experiment.position)))
       if length(tip_pos)==1 & abs(tip_pos)<=size(experiment.atom,2) then
           experiment.position=abs(round(tip_pos));
           c=%T;
       elseif length(tip_pos)==size(experiment.atom,2) then
           experiment.position=tip_pos;
           c=%T;
       end
    end
    
    show_window(cm_gui.figure_id);
    plt_coupling();
    if experiment.rate_calc>0 then
       cleardata(1);   //clear all calculated data except the Eigenvalues and -energies
    end
    
endfunction


function change_coupling(a)
    
   //this function allows to set for
   //a=1, the symmetric or asymmetric Heisenberg coupling constants
   //a=2, the Dzyaloshinsky-Moriya coupling vector
   //a=3, the scattering phase and amplitude between sites
      
   global experiment cm_gui hh;
   nbr=size(experiment.atom,2); //number of spins
   row="Spin: ";labelv=row(ones(1,nbr-1))+string(1:nbr-1);
   col="Spin: ";labelh=col(ones(1,nbr-1))+string(2:nbr);
   if a==1 then
       str=experiment.matrix;
       diagtxt="Heisenberg coupling between two spins";
   elseif a==2 then
       str=experiment.matrixDM;
       diagtxt="Dzyaloshinsky-Moriya coupling between two spins";
   elseif a==3 then
       str=experiment.Phase;
       diagtxt="Phase factor between two spins";
   else
       disp("ERROR, parameter out of range");
       abort;
   end
   
   for i=1:size(str,2)-1,
       str(i+1:$,i)="do not edit";
   end;
   
   err=1;
   while err<>0,
       new=x_mdialog(diagtxt,labelv,labelh,str);
       //check for valid expression        
       for i=1:nbr-1,
           for j=i:nbr-1,
               err=execstr('evstr(new(i,j))','errcatch');
               if err <> 0 then 
                   disp('Cell '+string(i)+', '+string(j)+' malformated'); 
                   i=nbr; 
                   j=nbr;
               end;
           end;
       end; 
   end;
   
   if size(new)==size(str) then
       if or(new<>str) then
           if a==1 then
               new(find(new=='do not edit'))='0';
               experiment.matrix=new;
               cleardata(2);
           elseif a==2 then
               new(find(new=='do not edit'))='0, 0, 0';
               experiment.matrixDM=new;
               cleardata(2);
           elseif a==3 then
               new(find(new=='do not edit'))='0';
               experiment.Phase=new;
               cleardata(1);
           else
               abort;
           end;
       end;
   end;
   
   show_window(cm_gui.figure_id);
   plt_coupling();
       
endfunction

function gui_drawstates(optional)
    
    //Optional parameter can be given to change either "B"-field or 
    //"C"-interaction
    
    global main_gui gui experiment calc;
    
    [lhs,rhs]=argn(0);
    
    if sum(experiment.B.^2)==0 &..
        (experiment.heisenberg_coupling==0 | size(experiment.atom,2)==1) then
        messagebox("Nothing to do!");
        abort;
    end
    
    plotdrawn=0;
    if calc.states==[] then
        //calculate the number of eigenstates
        nbr=1;
        atomlink=experiment.atom;
        for i=1:size(atomlink,2),
            nbr=nbr*(2*atomlink.S(i)+1);
        end
        answ=[];
        
        if nbr>1000 then
            msg=["The Hamiltonian has more than "+string(nbr)+" Eigenstates!";..
            "Do you really want to calculate that beast?"];
            answ=messagebox(msg, "modal", "info", ["Yes" "No"]);
        end
        if answ==2 then 
            abort;
        end
    
        if size(atomlink,2)<2 then
            winH=waitbar(0,'Wait for plotting states');
            scf(main_gui);
            [x,y]=Plotstates(atomlink,experiment.B);
            plotdrawn=1;answ=1;
            waitbar(1,winH);
            close(winH);
        else
            if rhs==0 then
                answ=messagebox("What shall be varied?", "modal","info", ["B-field" "Coupling (C)"]);
            else
                if optional=="B" then 
                    answ=1;
                else
                    answ=2;
                end
            end
                        
            if answ==0 then 
                abort;
            end
            
            if answ==1 then

                //vary the magnetic field
                HH=spzeros(nbr,nbr);
                ////
                //the heisenberg coupling matrix
                JH=experiment.matrix;
                JH1=experiment.heisenberg_coupling;
            
                for i=1:size(JH,1),
                    for j=2:size(JH,2)+1,
                        if j>i then
                            HH=HHeisenberg(atomlink,evstr(JH(i,j-1))*JH1,i,j)+HH;
                        end
                    end
                end
                ////
                ////
                //the DM coupling
                M=experiment.matrixDM;
                J=experiment.heisenberg_coupling;
        
                for i=1:size(M,1),
                    for j=2:size(M,2)+1,
                        if j>i then
                            HH=HDM(atomlink,evstr(M(i,j-1))*J,i,j)+HH;
                        end
                    end
                end
        
                b=linspace(0,1,experiment.no_eval);
                x=b*sqrt(sum(experiment.B.^2));
                clear y;
                winH=waitbar(0,'Wait for plotting states');
                gui_startcalc();
                if size(HH,1) < experiment.max_no_eigenstates+2 then
                    for i=1:experiment.no_eval,
                        H=full(HAni(atomlink,b(i)*experiment.B)+HH);
                        y(:,i)=real(spec(H)); //remove problems making imaginary part
                        if %zeroadj then
                            y(:,i)=y(:,i)-min(y(:,i));
                        end
                        if ~%stopcalc then
                            waitbar(i/experiment.no_eval,winH);
                        else
                            close(winH);
                            gui_stopcalc();
                            abort;    
                        end
                    
                    end,
                else
                    for i=1:experiment.no_eval,
                        H=HAni(atomlink,b(i)*experiment.B)+HH;
                         ofst=0;ofstold=0;
                         while ofst >= 0
                            H=H-ofst*speye(H); /////
                            [vcn,Cn]=eigs(H,speye(H),experiment.max_no_eigenstates);
                            ofstold=ofstold+ofst;
                            ofst=max(real(diag(vcn)));
                         end
                        y(:,i)=diag(real(vcn))+ofstold;
                        y=gsort(y,'r','i');
                        if %zeroadj then
                            y(:,i)=y(:,i)-min(y(:,i));
                        end
                        if ~%stopcalc then
                            waitbar(i/experiment.no_eval,winH);
                        else
                            close(winH);
                            gui_stopcalc();
                            abort;    
                        end
                    end,
                end,
                close(winH);
                gui_stopcalc("OK");
            else
                //vary the coupling C (experiment.heisenberg_coupling)
                HH=spzeros(nbr,nbr);
                ////
                //the heisenberg coupling matrix
                JH=experiment.matrix;
            
                for i=1:size(JH,1),
                    for j=2:size(JH,2)+1,
                        if j>i then
                            HH=HHeisenberg(atomlink,evstr(JH(i,j-1)),i,j)+HH;
//                            HH=Hdd(atomlink,evstr(JH(i,j-1)),i,j)+HH;
                        end
                    end
                end
                ////
                ////
                //the DM coupling
                M=experiment.matrixDM;
        
                for i=1:size(M,1),
                    for j=2:size(M,2)+1,
                        if j>i then
                            HH=HDM(atomlink,evstr(M(i,j-1)),i,j)+HH;
                        end
                    end
                end
                
                //the ansotropy and B-field matrix
                HA=HAni(atomlink,experiment.B);
        
                b=linspace(0,1,experiment.no_eval);
                x=b*experiment.heisenberg_coupling;
                clear y;
                winH=waitbar(0,'Wait for plotting states');
                gui_startcalc();
                if size(HH,1) < experiment.max_no_eigenstates+2 then
                    for i=1:experiment.no_eval,
                        H=full(HA+HH*x(i));
                        y(:,i)=real(spec(H)); //remove problem making imaginary part
                        if %zeroadj then
                            y(:,i)=y(:,i)-min(y(:,i));
                        end
                        if ~%stopcalc then
                            waitbar(i/experiment.no_eval,winH);
                        else
                            close(winH);
                            gui_stopcalc();
                            abort;
                        end
                    
                    end,
                else
                    for i=1:experiment.no_eval,
                        H=HA+HH*x(i);
                         ofst=0;ofstold=0;
                         while ofst >= 0
                            H=H-ofst*speye(H); /////
                            [vcn,Cn]=eigs(H,speye(H),experiment.max_no_eigenstates);
                            ofstold=ofstold+ofst;
                            ofst=max(real(diag(vcn)));
                         end
                        y(:,i)=diag(real(vcn))+ofstold;
                        y=gsort(y,'r','i');
                        if %zeroadj then
                            y(:,i)=y(:,i)-min(y(:,i));
                        end
                        if ~%stopcalc then
                            waitbar(i/experiment.no_eval,winH);
                        else
                            close(winH);
                            gui_stopcalc();
                            abort;
                        end
                    end,
                end,
                close(winH);
                gui_stopcalc("OK");
            end;
        end
       
        //shorten eventually x when interrupted
        ofs=min(y);
        y=gsort(y-ofs,'r')+ofs;
        calc.states=[x',y'];
    end
    
    scf(main_gui);
    if plotdrawn==0 then
        plot(calc.states(:,1),calc.states(:,2:$));
    end
    if exists("answ") then
        if answ==1 then
            gui.graph_ui.x_label.text="B (T)";
        else
            gui.graph_ui.x_label.text="C";
        end
    else
        gui.graph_ui.x_label.text="";
    end
    gui.graph_ui.y_label.text="E (meV)";
    gui.graph_ui.data_bounds=[min(calc.states(:,1)), min(calc.states(:,2: $))..
    ;max(calc.states(:,1)),max(calc.states(:,2:$))];
 
    
endfunction

function gui_eigenstates()
    
   global experiment calc;
    
   if experiment.Eigenval==[] then
      eigenvalues(experiment);
   end;
   
   [xsize,ysize]=size(experiment.Eigenvec);
   if calc.occ==[] then
       txt=["Colorplot of the states" "Reduced to the giant spin model",..
       "Matrix of Eigenvectors","Negativity"];
   else
       txt=["Colorplot of the states" "Reduced to the giant spin model",..
       "Matrix of Eigenvectors", "Negativity", "Occupation of the states",..
       "Lifetime (T1) of the states","Entropy of the spin system"];
       if %record_trans_matrix then
           txt=[txt "Bloch-Redfield coefficients"];
       end
   end
   
   answ=messagebox("Choose results to show", "modal", "question", txt);
    
   select answ
   
   case 1 then  //Colorplot of the states
   
       hh=scf();
       temp=hotcolormap(100); //black->red->white
       //temp2=[temp(:,3),temp(:,2),temp(:,1)]; //black->blue->white
       //temp2=temp2($:-1:1,:); //white->blue->black
       //temp=[temp2;temp]; // white->blue->black->red-white

       hh.color_map=temp;//temp($:-1:1,:); //invert colorscale
       
       maximum=max(abs(experiment.Eigenvec));
       colorbar(0,maximum);
       hh.children(1).y_label.text="absolute state prefactor";
       Matplot(0+(abs(experiment.Eigenvec(:,$:-1:1)')/maximum)*99.9,"081");
       //create x-label
       label=[];
       if  xsize<200 then
           //only draw labels on x-coordinate if the size is reasonable small
           for i=1:xsize,
               str=string(State(experiment.atom,i)');
               label=[label;strcat(['$\begin{bmatrix}',strcat(str(1:$),"\\"),'\end{bmatrix}$'])];
           end
           hh.children(2).x_ticks = tlist(["ticks", "locations", "labels"], (1:xsize)',label);
       end
       
       //create y-labels
       if ysize<200 then
           //only draw labels on y-coordinate if the size is reasonable small
           format(6);
           values=experiment.Eigenval-min(experiment.Eigenval);
           if max(values)<1 then
               str=string(values*1000);
               ylabel="$\mu\mbox{eV}$";
           else
               str=string(values);
               ylabel="$\mbox{meV}$"
           end
           format(10);
           label='$'+str+'$';
           hh.children(2).y_ticks = tlist(["ticks", "locations", "labels"], (1:ysize)',str);
           hh.children(2).y_label.text=ylabel;
       end    
       
       hh.figure_name="3rd order tunneling simulator. (c) Markus Ternes"
   
   case 2 then //Reduced to the giant spin model
   
       operatorS2=S2n(experiment.atom);
       answ2=messagebox("Choose quantization axis", "modal", "info",..
       ["X", "Y", "Z","B-field"]);
       select answ2
       case 1 then
          operatorSz=Sxn(experiment.atom);axistext="S_X";
       case 2 then
          operatorSz=Syn(experiment.atom);axistext="S_Y";
       case 3 then
          operatorSz=Szn(experiment.atom);axistext="S_Z";
       case 4 then
          operatorSz=Sarbn(experiment.atom,experiment.B(1),experiment.B(2),experiment.B(3));
          axistext="S_{\vec{B}}";
       else
          abort;
       end

       answ2=messagebox("Sort by:", "modal", "info",..
       ["S_total", axistext]);
       
       label=[];nSt=[];nSz=[];label2=[];nSt2=[];nSz2=[];
       for i=1:ysize,
           nS2=experiment.Eigenvec(:,i)'*operatorS2*experiment.Eigenvec(:,i);
           nSt2(i)=(sqrt(4*nS2+1)-1)/2;
           nSt(i)=round(nSt2(i)*10)/10;
           nSz2(i)=experiment.Eigenvec(:,i)'*operatorSz*experiment.Eigenvec(:,i);
           nSz(i)=round(nSz2(i)*1000)/1000;
           
       end
       nSz=nSz+10000; //to trick out gsort which will sort only abs()
       
       select answ2
       case 1 then          
           [nSt,k]=gsort(nSt); //sort according to largest total Spin:
           nSz=nSz(k); //reorder
           nSz2=nSz2(k);
           nSt2=nSt2(k);
           kk=k;
           vcn=experiment.Eigenval(k); //reorder
           for i=1:ysize,
               temp=find(nSt==nSt(i)); //find the S_tot which are the same
               [temp2,k]=gsort(nSz(temp));
               nSz(temp)=nSz(temp(k)); //reorder
               nSz2(temp)=nSz2(temp(k));
               //nSt(temp)=nSt(temp(k));
               nSt2(temp)=nSt2(temp(k));
               vcn(temp)=vcn(temp(k));
               kk(temp)=kk(temp(k));
               i=max(temp)+1;
           end
       case 2 then
           [nSz,k]=gsort(nSz); //sort according to largest Spin component (p.ex. S_z):
           nSt=nSt(k); //reorder
           nSz2=nSz2(k);
           nSt2=nSt2(k);
           kk=k;
           vcn=experiment.Eigenval(k); //reorder
           for i=1:ysize,   //TODO!!!
               temp=find(nSz==nSz(i)); //find the S_z which are the same
               [temp2,k]=gsort(nSt(temp));
               nSt(temp)=nSt(temp(k));  //reorder
               nSz2(temp)=nSz2(temp(k));
               //nSt(temp)=nSt(temp(k));
               nSt2(temp)=nSt2(temp(k));
               vcn(temp)=vcn(temp(k));
               kk(temp)=kk(temp(k));
               i=max(temp)+1;
           end
       else
          abort;
       end;        
       
              nSz=nSz-10000;
       format(5);
       for i=1:ysize,
           label=[label;'$\Psi_{'+string(kk(i))+'}=$\\ $\begin{bmatrix}'+string(nSt(i))+"\\"+string(nSz(i))+'\end{bmatrix}$'];
           label2=[label2;[vcn(i)-min(vcn),nSt2(i),nSz2(i)]];
       end;
       hh=scf();
       calc.giantspin=label2;
       calc.giantspin_label=['Energy','S_total', axistext];
       plot(vcn-min(vcn),'-b.');
       hh.children.x_ticks = tlist(["ticks", "locations", "labels"], (1:ysize)',label);
       hh.children.x_label.text="$\begin{bmatrix}\left< S_{total} \right> \\ \left<"+axistext+"\right> \end{bmatrix}$";
       hh.children.y_label.text="E (meV)";
       hh.children.margins=[0.125 0.125 0.125 0.30];
       format(10);
       xgrid(5);
       hh.figure_name="3rd order tunneling simulator. (c) Markus Ternes"
        
   case 3 then //Matrix of Eigenvectors
   
       Eigenvalues=experiment.Eigenval;
       Eigenvectors=experiment.Eigenvec;
        
       editvar('Eigenvectors');
       editvar('Eigenvalues');
       
   case 4 then //Negativity
       
       if size(experiment.atom,2) < 2 then
         messagebox("For entanglement you need at least two coupled spins");
         abort;
       end
       
       occ=Occupation(experiment.Eigenval,experiment.T);

       rho=full(densitymatrix(experiment));

       neg=negativity(rho,experiment.atom);
       xaxis=[1:size(experiment.atom,2)]';
       hh=scf();
       plot(xaxis,neg,'o-');
       xlabel("$\mbox{atom number}$");
       ylabel("$\mbox{negativity}$");
       calc.negativity=[xaxis,neg];
       hh.figure_name="3rd order tunneling simulator. (c) Markus Ternes";
       
   case 5 then //Occupation of the states
       
       hh=scf();
       plot2d('nn',calc.occ(:,1),calc.occ(:,2:$));
       if rate_calc==3 then 
          xlabel("$\mbox{time (ps)}$");
       else
          xlabel("$\mbox{eV (meV)}$");
       end
       
       ylabel("State occupation");

       if size(experiment.Eigenvec,2)<20 then
           legend("$|\Psi_{"+string(1:size(experiment.Eigenvec,2))+"}|^2$");
       end
       
       hh.figure_name="3rd order tunneling simulator. (c) Markus Ternes";

   case 6 then //Lifetime of the states
    
    hh=scf();
    plot2d('nl',calc.lifetime(:,1),calc.lifetime(:,2:$));
    if experiment.rate_calc==3 then 
        xlabel("$\mbox{time (ps)}$");
    else
        xlabel("$\mbox{eV (meV)}$");
    end
    ylabel("$\mbox{lifetime (s)}$");
    hh.figure_name="3rd order tunneling simulator. (c) Markus Ternes";
    
    if size(experiment.Eigenvec,2)<20 then
        legend("$\tau_{"+string(1:size(experiment.Eigenvec,2))+"}$");
    end
    
   case 7 then //Entropy of the spin system
       
    hh=scf();
    vnent=[];
    if calc.dm==[] then
        for i=1:size(calc.occ,1),
            //vnent(i)=entropy(diag(gui.user_data_occ(i,2:$)));
            tmp=-calc.occ(i,2:$).*real(log(calc.occ(i,2:$))); //avoid compex results
            tmp(isnan(tmp))=0;  //replace %nan with 0
            vnent(i)=sum(tmp);
        end
        xl="$\mbox{eV (mV)}$";
    else
        for i=1:size(calc.dm,3),
            vnent(i)=vNentropy(calc.dm(:,:,i));
        end
        xl="$\mbox{time (ps)}$";
    end;
    calc.entropy=[calc.occ(:,1),vnent];
    plot2d('nn',calc.occ(:,1),vnent);
    xlabel(xl);
    ylabel("$\mbox{entropy }(k_B)$");    
    
    hh.figure_name="3rd order tunneling simulator. (c) Markus Ternes";
    
   case 8 then //display Bloch-Redfield coefficients
       
    e=experiment.Eigenval-min(experiment.Eigenval);
    len=length(e);

    disp_state=[];
      
    if experiment.RedF_base=="energy" then
        txt=[];
        for i=1:len,
            txt=[txt; 'State '+string(i)+' @ '+string(e(i))+' meV'];
        end
                
        sel_1=x_choose(txt,['Select column entry in the energy base of the density matrix']);
        sel_2=x_choose(txt,['Select row entry in the energy base of the density matrix']);
        if sel_1<>[] & sel_2<>[] then
            disp_state=[sel_1,sel_2];
        else
            abort;
        end
    else
        e=1;
        for i=1:size(experiment.atom,2),
            e=e*(2*experiment.atom(i).S+1);
        end
        txt=string(State(experiment.atom,1:e));
        txt=strcat(txt,' ','c');
            
        sel_1=x_choose(txt,['Select column entry in the spin base of the density matrix']);
        sel_2=x_choose(txt,['Select row entry in the spin base of the density matrix']);
        if sel_1<>0 & sel_2<>0 & sel_1<>sel_2 then
            disp_state=[sel_1,sel_2];
        else
            abort;
        end
    end
    scf();
    y=calc.trans_matrix(sel_1+(sel_2-1)*len:len^2:$,sel_1+(sel_2-1)*len);
   
    plot2d('nl',-real(y).^-1*1e-12); //rescale to (ps)
    xlabel("$\mbox{#}$");
    ylabel("$\mbox{lifetime (s)}$");
    legend("$\rho_{"+string(sel_1)+", "+string(sel_2)+"}$");
    
   calc.T2=[-real(y)^-1*1e-12];
   calc.T2=[(1:length(calc.T2))',calc.T2];
   
   end;
    
endfunction

function gui_limits()
    
    global minvalues1 maxvalues1 labels1;
    
    labelh=["min";"max"];

    if %Stevstr<>"" then 
        maxsl=size(%Stevstr,2)+13;
    else
        maxsl=15;
    end
    
    val=string([minvalues1(1:maxsl);maxvalues1(1:maxsl)]');
    newval=x_mdialog('Set parameter boundaries',labels1(1:maxsl),labelh,val);
    
    if newval<>[] & ~and(newval==val) then
       newval(~isnum(newval))=val(~isnum(newval)); //replace bullshit and set it to old value  
       minvalues1(1:maxsl)=evstr(newval(:,1))';
       maxvalues1(1:maxsl)=evstr(newval(:,2))';
    end
    update_slider();
    
endfunction

function gui_del()
    
    global gui;
    if gui.graph_ui.visible=='off' then
        gui.graph_ui.visible='on';
    else
        if gui.graph_ui.children<>[] then
           delete(gui.graph_ui.children(1))
        end
    end
    
endfunction

function gui_del_all()
    
    global gui %Data Data_fname;
    gui.graph_ui.visible='on';
    nbrofplts=size(gui.graph_ui.children,1);
    if gui.graph_ui.children<>[] then
       gui.graph_ui.data_bounds=[0,0;0,0];
       delete(gui.graph_ui.children);
    end
    if size(%Data,1)>1 & nbrofplts>2 then
      scf(main_gui);
      nr=size(%Data,2)-1;
      plot2d(%Data(:,1),%Data(:,2:$),linspace(-9,-9,nr));
      gui.graph_ui.children(1).children.mark_foreground=2;
      gui.graph_ui.children(1).children.mark_size_unit='point';
      gui.graph_ui.children(1).children.mark_size=4;
      captions(gui.graph_ui.children(1).children(1), Data_fname);
    elseif size(%Data,1)>1 then
      mssg=messagebox("Do you really want to remove the experimental data?"..
      ,"modal", "question", ["Yes" "No"]);
      if mssg==1 then
          %Data=[];
      else
          scf(main_gui);
          nr=size(%Data,2)-1;
          plot2d(%Data(:,1),%Data(:,2:$),linspace(-9,-9,nr));
          gui.graph_ui.children(1).children.mark_foreground=2;
          gui.graph_ui.children(1).children.mark_size_unit='point';
          gui.graph_ui.children(1).children.mark_size=4;
          captions(gui.graph_ui.children(1).children(1), Data_fname);  
      end
    end
    
endfunction

function gui_save(name,typ)
    //if only "name" is given, then ASCII Data is saved in the file "name"
    //we will always save the first entry in calc. except "typ" is given, 
    //i.e. gui_save("file.dat",2) would save ASCII of "States versus B/C"
    //if typ="SOD" then parameters are saved in "name"
    
    global gui experiment %Data scrpt;
    
    [lhs,rhs]=argn(0);
    
    if rhs==0 then
        answ=messagebox("What do you want to save", "modal", "info",..
       ["Project" "Calculation" "Script"]);
    elseif rhs==1 then  //if name is only given then save calculation
        answ=2;
    elseif typ=="SOD" then
        answ=1;
    end
    
    select answ
        case 1 then //save project
            if rhs==0 then
                filename=uiputfile("*.SOD");
            else
                filename=name;
            end
        
            if filename<>"" then
                save(filename,'experiment','calc');
            end
            
        case 2 then     //save calculation
            //create header text:
            text=[];    
            fieldn=fieldnames(experiment);
            for i=1:size(fieldn,1),
                if fieldn(i)<>"Eigenvec" & fieldn(i)<>"Eigenval"..
                     & fieldn(i)<>"matrix" & fieldn(i)<>"matrixDM".. 
                     & fieldn(i)<>"Phase".. 
                     & fieldn(i)<>"atom" & fieldn(i)<>"RedF_coeff"..
                     & fieldn(i)<>"ptipK" & fieldn(i)<>"psampleK"..
                     & fieldn(i)<>"RedF_dm_init" & fieldn(i)<>"RedF_dm_calc_list"
                     then
                   execstr("tmp=experiment."+fieldn(i));
                   if tmp==[] then 
                       text=[text;fieldn(i)+"=[]"];
                   else
                       text=[text;fieldn(i)+"="+strcat(string(tmp),";")];
                   end
                   
                end;
            end;     
            atomlink=experiment.atom;
            
            
            if size(atomlink,2)>1 then
               tmp=experiment.matrix;
               text=[text;' ';'Heisenberg Matrixvalues:']
               for i=1:size(tmp,2),
                   str=' ';
                   for j=1:size(tmp,1),
                       str=str+tmp(i,j)+';';
                   end
                   text=[text;str];
               end
               tmp=experiment.matrixDM;
               text=[text;' ';'DM Matrixvalues:'];
               for i=1:size(tmp,2),
                   str=' ';
                   for j=1:size(tmp,1),
                       str=str+tmp(i,j)+';';
                   end
                   text=[text;str];
               end
               tmp=experiment.Phase;
               text=[text;' ';'Phase factors:'];
               for i=1:size(tmp,2),
                   str=' ';
                   for j=1:size(tmp,1),
                       str=str+tmp(i,j)+';';
                   end
                   text=[text;str];
               end
            end
            
            for i=1:size(atomlink,2)
                text=[text;" ";"Spin "+string(i)+":"];
                fieldn=fieldnames(atomlink(i));
                for j=1:size(fieldn,1),
                   execstr("tmp=atomlink("+string(i)+")."+fieldn(j));
                   text=[text;fieldn(j)+"="+strcat(string(tmp),";")];
                end;
            end;
            
            dataexist=[];
            for i=1:13,
               dataexist(i)=getfield(i+2,calc)<>[];
            end
        
            labelpos=['Spectrum','States versus B or C','Occupation of States',..
            'Lifetime of States','Entropy of System','Giant Spin Approx.',..
            'Negativity','Magnetization','Class. Correlation','QM Correlation',..
            'Mean of Spectra','Multidata','T2'];
        
            label=labelpos(dataexist);
            
            if calc.dm<>[] then //add density matrix to it if exist
                label=[label,'Density Matrices'];
            end
                        
            if label==[] then 
                messagebox("No calculated data to save");
                abort;
            end
        
            if rhs==0 then
                answ=messagebox("Which data do you want to save", "modal", "info",label);
                if answ==0 then 
                    abort; 
                end; 
            
            elseif rhs==1 then
                answ=1;
            else
                answ=typ;
            end;
            
            if label(answ)<>"Density Matrices" then  //it's not the density matrices
                pos=find(dataexist==%T);
                data=getfield(pos(answ)+2,calc); //extract data
        
                if rhs==0 then
                    filename=uiputfile("*.dat");
                else
                    filename=name;
                end;
        
                if filename<>"" then
                    savedat(filename,data($:-1:1,1),real(data($:-1:1,2:$)),text);
                end;
            else //it's the density matrices
                if rhs==0 then
                    filename=uiputfile("*.dat");
                else
                    filename=name;
                end;
        
                if filename<>"" then
                    str=["Density Matrix (DM) Data calculated with a routine written by";..
                    "Markus Ternes (markus.ternes@gmail.com)";" "; "Initial DM below:"];
                    print(filename,calc.dm,experiment.RedF_dm_init,str);
                    fd = mopen(filename, 'rt');
                    str1 = mgetstr(1e9, fd);
                    mclose(fd);
                    for i=1:size(calc.dm,3),
                        search_str="(:,:,"+string(i)+")";
                        replace_str=ascii(10)+"Next DM is No:"+string(i)+" at t="+..
                        string(experiment.RedF_dm_calc_list(i,1))+" ps, V="+..
                        string(experiment.RedF_dm_calc_list(i,2))+" mV, MW="+..
                        string(experiment.RedF_dm_calc_list(i,3))+" GHz, P="+..
                        string(experiment.RedF_dm_calc_list(i,4))+" rad/ps";
                        str1=strsubst(str1,search_str,replace_str);
                    end
                    fd = mopen(filename, 'wt');
                    mputstr(str1, fd);
                    mclose(fd);
                    
                        
                end;
            end
            
        case 3 then     //save script
            filename=uiputfile("*.sci");
            
            if filename<>"" then
                fd = mopen(filename,'wt');
                mputl(scrpt,fd);
                err=mclose(fd);
            end
        end;
    
endfunction

function err=gui_load(name,typ)    
    
    //if only "name" is provided, then ASCII Data is loaded from that file
    //if additionally "typ" is provided, then either ASCII or SOD can be loaded 
    //if no parameter is given, then the menu opens. 
    
    global main_gui %Data guientry1 guicheckbox1 experiment gui PATH Data_fname calc;
    global scrpt;
    
    [lhs,rhs]=argn(0);
    
    if rhs==0 then
        if exists('IBM_import') then
            answ=messagebox("What do you want to load", "modal", "info",..
            ["Project" "ASCII Data" "Script" "old IBM Data" "QNS mat Data"]);
        else
            answ=messagebox("What do you want to load", "modal", "info",..
            ["Project" "ASCII Data" "Script"]);
        end
    elseif rhs==1 then
        answ=2; //load ASCII data
    elseif typ=="SOD" then
        answ=1;  //load SOD file
    elseif typ=="mat" then
        answ=5; //load mat data
    else
        answ=2;  //load ASCII data
    end 
    
    select answ
    case 1 then
       
       if rhs<>0 then           //filename is given in the parameters
           filename=name; filedir="";
       else                     //filename has to be given via gui 
           [filename,filedir]=uigetfile(["*.sod";"*.SOD"],PATH);
       end
       
       if filename<>"" then
           cleardata(2); //delete all data
           backupcalc=calc;    //backup an empty calc variable
           backup_use_Stevens=experiment.atom(1).use_Stevens;  
           //backup of the Stevens switch
           
           gui_last_spin=1; //set it back to the first spin
           
           err=execstr("load(filedir+''/''+filename,''experiment'',''calc'')",'errcatch');
           
           if err then
               // Try if it is a old SOD file prior version 0.991
               err=execstr("load(filedir+''/''+filename,''experiment'',''atomlink'')",..
               'errcatch');
               if ~err then
                   disp('Old SOD file prior version 0.991')
                   //check if DM interaction is included to be compatible
                   if ~or(getfield(1,experiment)=='matrixDM') then
                       experiment.matrixDM='0, 0, 0';
                   end;
               else
                   disp('Reading file error')
               end
           end;
           
           txt=getfield(1,experiment);
           
           //check if paramag_S is included to be compatible
           if grep(txt,'paramag_S')==[] then
               experiment.paramag_S=5/2;
               experiment.eta=0.3;
               experiment.paramag_g=2;
               experiment.paramagnetic=%F;
               disp('Old SOD file. '+..
               'Options for paramagnetic tip missing. Standart values will be used.')
           end
            
           //check if func_tip is included to be compatible
           if grep(txt,'func_tip')==[] then
              experiment.func_tip=0;
              experiment.corr_sample=%f;
              experiment.corr_tip=%f;
              disp('Old SOD file. '+..
              'Option for functionalized tip missing. Standart value will be used.')
           else
              if experiment.func_tip==%T then
                experiment.func_tip=1;
              end
              if grep(txt,'corr_tip')==[] then
                  experiment.corr_tip=%f;
              end
              if grep(txt,'psampleK')==[] then
                  experiment.psampleK=[];
                  experiment.ptipK=[];
              end             
           end
           
           //check if experiment.spectrum_deriv exists to be compatible           
           if grep(txt,'spectrum_deriv')==[] then
               experiment.spectrum_deriv=1;
               disp('Old SOD file. '+..
               'Option for different derivatives missing. Standart value will be used.')               
           end
 
           //check if experiment.atom.Stevens exists to be compatible
           txtatom=getfield(1,experiment.atom(1));
           if grep(txtatom,'use_Stevens')==[] then
               disp('Old SOD file. '+..
               'Option for Stevens operators is missing. Standart value will be used.')               
               for i=1:size( experiment.atom,2),
                   experiment.atom(i).use_Stevens=0;   //set to zero
                   experiment.atom(i).Stevens=[0];
               end
               //adapt size of experiment.sw to be compatible
               if length(experiment.sw)>13 then 
                   experiment.sw(5:13)=experiment.sw(6:14); 
                   //to account for version 0.999
               end
               experiment.sw(length(experiment.sw)+1:13)=%T;
               experiment.sw=experiment.sw(1:13);
           end
           
           
           if grep(txtatom,'alpha')==[] then
                disp('Old SOD file. '+..
                'Options for the angles alpha, beta, and gamma are missing. Standart values will be used.')     
                for i=1:size( experiment.atom,2),
                    experiment.atom(i).alpha=0;   //set to zero
                    experiment.atom(i).beta=0;
                    experiment.atom(i).gamma=0;
                end
                l=length(experiment.sw);
                experiment.sw(l-7:l+3)=experiment.sw(l-10:l); //adjust size
           end

           if or(size(experiment.atom(1).Stevens)<>[3 7]) then
               for i=1:size(experiment.atom,2),
                   experiment.atom(i).Stevens(3,6:7)=0;
               end
               disp('Old SOD file. Size of Stevens operators wrong. Adjust size.');               
           end
           
           //check if experiment.spec_length exists to be compatible (verison 1.6)
           if grep(txt,'spec_length')==[] then
               experiment.spec_length=1000;
               disp('Old SOD file.'+..
               'Option for spectrum length is missing. Standart value will be used.')               
           end
           
           //check if experiment.use_T_ts exists to be compatible (verison 1.6)
           if grep(txt,'use_T_ts')==[] then
               experiment.use_T_ts=%F;
               experiment.T_ts=[1 1 1];
               disp('Old SOD file.'+..
               'Option for tip-sample scattering parameters are missing. '+..
               'Standart values will be used.')       
           end
           
           //check if the Bloch-Redfield parameters are included to be copatible
           if grep(txt,'use_T_ts')==[] & %Bloch_Redfield then
               experiment.RedF_t_total=100000;
               experiment.RedF_V_pulses=[0,5000];
               experiment.RedF_V_amplitudes=10;
               experiment.RedF_V_Gauss_Noise=0;
               experiment.RedF_MW_pulses=[];
               experiment.RedF_MW_amplitudes=0;

               experiment.RedF_secular_window=0.01;
               experiment.RedF_secular="window";
               experiment.RedF_dm_init=[]; 
               experiment.RedF_dm_calc_list=[];
               experiment.RedF_base="energy";
               disp('Old SOD file.'+..
               'Bloch-Redfield  parameters are missing. '+..
               'Standart values will be used.')       
           end
           
            //check if DM interaction is included to be compatible
           if  grep(txt,'Phase')==[] then
               experiment.Phase='0';
                disp('Old SOD file.'+..
               'Phase factors are missing. '+..
               'Standart values will be used.')       
           end;
           
           
           
           PATH=filedir;
            
           backupcalc(:)=calc; //store so calculations aren't deleted
           if experiment.atom(1).use_Stevens<>backup_use_Stevens then
               build_main_gui();
           else
               for k=1:13 //length(experiment.sw),
                   guicheckbox1(k).value=1*(~experiment.sw(k));
               end
           
               update_gui_values();
               number_spins_callback();
               checkbox1_callback();
           
           end
           calc=backupcalc;
       else
           abort();
       end
    
    case 2 then  
       if rhs<>0 then
           Data_fname=name;
           filedir="";
       else
           [Data_fname,filedir]=uigetfile("*.dat",PATH);
       end
       
       if Data_fname<>"" then
           fn=mopen(filedir+'/'+Data_fname);
           txt = mgetl(fn);
           dat=[];
           PATH=filedir;
           for i=1:size(txt,1),
               [a,err]=evstr(txt(i));
               if err==0 then
                  dat=[dat;a];
               end,
           end
           err=mclose(fn);
           if size(dat,2)> 2 & rhs==0 then
              text=["Data has more than two columns.";..
              "Extracting Data from column "+..
               gui.sel_Load_Column.String+" ?"];
               answ2=messagebox(text, "modal", "info",..
               ["Yes" "No"]);
               if answ2==1 then
                   if abs(evstr(gui.sel_Load_Column.String))>size(dat,2) then
                       gui.sel_Load_Column.String=..
                       string(sign(evstr(gui.sel_Load_Column.String))*size(dat,2));
                   end
                
                   %Data=[1000*dat(:,1),dat(:,abs(evstr(gui.sel_Load_Column.String)))];
                   if evstr(gui.sel_Load_Column.String)<0 then
                       %Data(:,2)=-%Data(:,2); //invert the data
                   end;
               else
                       %Data=[dat(:,1),dat(:,2:$)];
               end
           elseif size(dat,2)==2 then
               %Data=[dat(:,1),dat(:,2)];
           else
               %Data=[1000*dat(:,1),dat(:,abs(evstr(gui.sel_Load_Column.String)))];
               if evstr(gui.sel_Load_Column.String)<0 then
                   %Data(:,2)=-%Data(:,2); //invert the data
               end;
           end;
           scf(main_gui);
           nr=size(%Data,2)-1;
           plot2d(%Data(:,1),%Data(:,2:$),linspace(-9,-9,nr));

           gui.graph_ui.children(1).children.mark_foreground=2;
           gui.graph_ui.children(1).children.mark_size_unit='point';
           gui.graph_ui.children(1).children.mark_size=4;
           captions(gui.graph_ui.children(1).children(1), Data_fname);
            
           if %Stevstr<>"" then 
               maxsl=size(%Stevstr,2)+17;
           else
               maxsl=19;
           end
            
           if nr==1 then
              guientry1(maxsl).String=string(max(abs(%Data(:,1))));   //bias 
           else
              guientry1(maxsl-3).String=string(max(abs(%Data(:,1))));   //Field in Bz
              guientry1(maxsl-2).String="0"; 
              guientry1(maxsl-1).String="0";
           end
           entry1_callback();
           //update_gui_values();
           //update_atomlink();
       end
       err=0;
   case 3 then
       [filename,filedir]=uigetfile(["*.sci";"*.SCI"],PATH);
       if filename<>"" then
           fn=mopen(filedir+'/'+filename);
           scrpt = mgetl(fn);
           err=mclose(fn);
       else
           abort();
       end           
        
   case 4 then  
       [Data_fname,filedir]=uigetfile("*.*",PATH);
       if filename<>"" then
           [%Data,Header]=IBM_import(filedir+'/'+Data_fname);
           %Data(:,1)=%Data(:,1)*1000; //rescale to mV
           scf(main_gui);
           PATH=filedir;
           plot2d(%Data(:,1),%Data(:,2)),-9;
            
           if %Stevstr<>"" then 
               maxsl=size(%Stevstr,2)+17;
           else
               maxsl=19;
           end

           guientry1(maxsl).String=string(max(abs(%Data(:,1))));
           gui.graph_ui.children(1).children.mark_foreground=2;
           gui.graph_ui.children(1).children.mark_size_unit='point';
           gui.graph_ui.children(1).children.mark_size=4;
           entry1_callback();
           //update_gui_values();
           //update_atomlink();
           err=0;
       else
           abort();
       end;
       
   case 5 then  
       
       if rhs<>0 then
           Data_fname=name;
           filedir="";
       else
           [Data_fname,filedir]=uigetfile("*.mat",PATH);
       end
       
       if Data_fname<>"" then
           [%Data]=mat_import(filedir+'/'+Data_fname,abs(evstr(gui.sel_Load_Column.String)));

           if evstr(gui.sel_Load_Column.String)<0 then
               %Data(:,2)=-%Data(:,2); //invert the data
           end;

           %Data(:,1)=%Data(:,1)*1000; //rescale to mV
           scf(main_gui);
           PATH=filedir;
           
           scf(main_gui);
           nr=size(%Data,2)-1;
           plot2d(%Data(:,1),%Data(:,2:$),linspace(-9,-9,nr));

           gui.graph_ui.children(1).children.mark_foreground=2;
           gui.graph_ui.children(1).children.mark_size_unit='point';
           gui.graph_ui.children(1).children.mark_size=4;
           captions(gui.graph_ui.children(1).children(1), Data_fname);
           
           if %Stevstr<>"" then 
               maxsl=size(%Stevstr,2)+17;
           else
               maxsl=19;
           end
           guientry1(maxsl).String=string(max(abs(%Data(:,1))));
           entry1_callback();
           //update_gui_values();
           //update_atomlink();
           err=0;
       else
           abort();
       end;
   else
       err=0;
   end
endfunction

function udata(S)
    
    global experiment ;

    number=gui.sel_spin.Value;
    experiment.atom(number).S=S.S;
    experiment.atom(number).g=S.g;
    experiment.atom(number).D=S.D;
    experiment.atom(number).E=S.E;
    experiment.atom(number).U=S.U;
    experiment.atom(number).J=S.J;
    experiment.atom(number).w=S.w;
    
    if experiment.atom(number).use_Stevens <> 0 then
        experiment.atom(number).use_Stevens=0;
        build_main_gui(); 
    end
    
    cleardata(2);
    update_gui_values();
    update_atomlink();

    //check if figure ani_gui exist
    ierr = execstr("delete(ani_gui.children)", "errcatch"); 
    
    if ~ierr then
       gui_disable();
       gui_checkanisotropy(); //yes then update figure
       gui_enable();
    end
    
endfunction

function gui_parameters()
    
    global experiment gui %savedata %plotspec %precission %normalize %zeroadj %renormalize_E;
    
    txt = ['Max # of eigensstates';'Gamma (meV)';'Min state occupancy';..
    '# of evaluations';'Length of spectrum';'Save all data immediatelly';..
    'Plot all spectra';'Normalize data';'Adjust lowest state to zero';..
    'Renormalize energies';'Use T_ts instead of coupling amplitude T^2'];
    sig=[string(experiment.max_no_eigenstates);string(experiment.lt);..
    string(%precission);string(experiment.no_eval);..
    string(experiment.spec_length);string(%savedata);..
    string(%plotspec);string(%normalize);string(%zeroadj);..
    string(%renormalize_E);string(experiment.use_T_ts)];
    
    answ = x_mdialog('Modify some global parameter. BE CAREFULL!',txt,sig);
    if answ==[] then
        answ=sig;
    end
    answ(~isnum(answ(1:5)))='1'; //replace bullshit

    if experiment.max_no_eigenstates <> evstr(answ(1)) then
        experiment.max_no_eigenstates = evstr(answ(1));
        cleardata(2);
    end;
    
    if experiment.lt <> evstr(answ(2)) then
        experiment.lt = evstr(answ(2));
        cleardata(1);
    end
    
    if %precission <> evstr(answ(3)) then
        %precission = evstr(answ(3));
        cleardata(1);
    end
    
    if experiment.no_eval <> evstr(answ(4)) then
        experiment.no_eval = evstr(answ(4));
        cleardata(2);
    end

    if experiment.spec_length <> evstr(answ(5)) then
        experiment.spec_length = evstr(answ(5));
        cleardata(1);
    end
    
    old=%savedata;
    if answ(6)=='%T'|answ(5)=='%t'|answ(6)=='T' then 
        %savedata=%T
    else
        %savedata=%F;
    end
    if old<>%savedata then cleardata(1),end;
        
    old=%plotspec;    
    if answ(7)=='%T'|answ(7)=='%t'|answ(7)=='T' then 
       %plotspec=%T; 
    else
       %plotspec=%F;
    end
    if old<>%plotspec then cleardata(1),end;
    
    old=%normalize;
    if answ(8)=='%T'|answ(8)=='%t'|answ(8)=='T' then 
       %normalize=%T
    else
       %normalize=%F;
    end
    if old<>%normalize then cleardata(1),end;
    
    old=%zeroadj;
    if answ(9)=='%T'|answ(9)=='%t'|answ(9)=='T' then 
       %zeroadj=%T;
    else
       %zeroadj=%F;
    end
    if old<>%zeroadj then cleardata(1),end;
    
    old=%renormalize_E; 
    if answ(10)=='%T'|answ(10)=='%t'|answ(10)=='T' then 
       %renormalize_E=%T;
    else
       %renormalize_E=%F;
    end
    if old<>%renormalize_E then cleardata(2),end;
    
    old=experiment.use_T_ts; 
    if answ(11)=='%T'|answ(11)=='%t'|answ(11)=='T' then 
       experiment.use_T_ts=%T;
    else
       experiment.use_T_ts=%F;
    end
    if old<>experiment.use_T_ts then 
        cleardata(1);
        experiment.A=1; //it's crucial to switch off the Amplitude !
        build_main_gui();
        sleep(100);    
         //check if figure ts_gui exist
        ierr = execstr("delete(ts_gui)", "errcatch"); 
        disp(ierr);
        if ~ierr then
            gui_disable();
            gui_tipsample(); //yes then update figure
            gui_enable();
        end
    end;
    
endfunction

function gui_switchexp()
    
    global experiment experiment2 guicheckbox1 guientry1 guislider1 gui calc;
    
    temp=experiment;
    experiment=experiment2;
    experiment2=temp;
    
    if experiment.atom(1).use_Stevens <> experiment2.atom(1).use_Stevens then
        build_main_gui()        
    end

    update_gui_values();
    number_spins_callback();
    
    cleardata(2);   //clear all calculation
    
endfunction

scrpt=["// simple script for producing multiplots by varying one parameter";..
"";..
"B=0:1:10;";..
"for i=B,         //start of loop";..
"   experiment.B=[0 0 i];     //change parameter";..
"   draw_spectrum();";..
"end";..
"";..
"x=calc.spec(:,1);";..
"y=B;";..
"z=multidat;";..
"f=scf();";"f.color_map = jetcolormap(256);";..
"grayplot(x,y,z);"]

function [scrpt]=gui_script(scrpt)

    global experiment guicheckbox1 guientry1 guislider1 gui ani_gui calc multidat;
    
    function draw_spectrum(vararg)
        global multidat calc

        [lhs,rhs]=argn(0); 
        
        update_gui_values();
        
        if rhs==0 then
            a=2;
        else
            a=vararg(1);
        end
        cleardata(a);
        gui_drawspec('silent');
        multidat=[multidat,calc.spec(:,gui.sel_spin.value+1)]
    endfunction
    
    function draw_states(vararg)
        global multidat calc
        
        [lhs,rhs]=argn(0); 
        
        update_gui_values();
        
        cleardata(2);
        gui_drawstates('silent');
        if rhs==0 then
            vararg=2;
        end
        multidat=[multidat,calc.states(:,[vararg])]
    endfunction
        
    answ=scrpt//
    //answ=x_dialog('Script for execution',scrpt);

    if answ<>[] then
        multidat=[];
        ierr=execstr(answ,'errcatch'); 
        scrpt=answ;
        if ierr==0 then
            calc.spec=[calc.spec(:,1),multidat];
        end
    end

endfunction


function gui_checkanisotropy()

    global experiment guicheckbox1 guientry1 guislider1 gui ani_gui;
    sel=findobj('Tag', 'sel_spin')
    number=sel.Value;
    use_Stevens=experiment.atom(number).use_Stevens;
    
//    ierr = execstr("close(ani_gui)", "errcatch");
    ierr = execstr("delete(ani_gui.children(:))", "errcatch");
    
    if ierr<>0 then
        ani_gui=figure('figure_size',[580,720],..
        'auto_resize','on',..
        'resize','off',..
        'figure_name',..
        '3rd order tunneling simulator. (c) Markus Ternes ');
    
        ani_gui.color_map = coolcolormap(128);
        ani_gui.background=-2
    end
    scf(ani_gui);
    hh.data=[];
    hh.graph_ui=newaxes();
    hh.graph_ui.margins=[0 0 0 0.0];
    hh.graph_ui.axes_bounds=[0.05,0.05,0.9,0.9];
  
    subplot(211);
    scf(ani_gui);
//    w=gcf();
    w.children(1).axes_bounds=[0 0 1 0.6];
    //drawanisotropy(S);
    S.S=experiment.atom(number).S;
    H=HAni(experiment.atom(number),experiment.B);
    tempEigenval=spec(H);

    //dm=Occupation(tempEigenval,experiment.T);
    dm=zeros(tempEigenval); dm(1,1)=1;
    
    scf(ani_gui);
    text="$\mbox{Visualization of the magnetic anisotropy of "
    if experiment.func_tip & number==1 then
        text=text+"the tip''s spin system}$"
    else
        text=text+"the sample spin No. "+string(number)+"}$";
    end
    drawanisotropy(H,diag(dm),text);
    //drawanisotropy(HAni(experiment.atom(number),experiment.B),nr);
    
    subplot(212);
    
    str="";
    if use_Stevens==0 | use_Stevens==%f then
        S.D=experiment.atom(number).D;
        S.E=experiment.atom(number).E;
        badDE=~(3*abs(S.E)<abs(S.D) & S.E>=0);
        SS=Dxyz(S);
        
        str=["$\mbox{The anisotropy parameters:}$";"$D="+string(S.D)+", E="+..
        string(S.E)+"$";"$\mbox{correspond to}$";"$D_{xx}="+string(SS.Dxx)+..
        ", D_{yy}="+string(SS.Dyy)+", D_{zz}="+string(SS.Dzz)+"$"];
        
        SN.Dxx=SS.Dyy;SN.Dyy=SS.Dxx;SN.Dzz=SS.Dzz;
        SN=Dde(SN);
        if 3*abs(SN.E)<abs(SN.D) & badDE then
            str=[str;"$\bf \quad\mbox{Flipping }x \rightleftharpoons y\mbox{ axes results in}$";..
            "$\bf D="+string(SN.D)+", E="+string(SN.E)+"\quad$"];
        else 
            str=[str;"$\quad\mbox{Flipping }x \rightleftharpoons y\mbox{ axes results in}$";..
            "$D="+string(SN.D)+", E="+string(SN.E)+"\quad$"];
    
        end
    
        SN.Dxx=SS.Dzz;SN.Dzz=SS.Dxx;SN.Dyy=SS.Dyy;
        SN=Dde(SN);
        if 3*abs(SN.E)<abs(SN.D) & badDE then
            str=[str;"$\bf \quad\mbox{Flipping }x \rightleftharpoons z\mbox{ axes results in}$";..
        "$\bf D="+string(SN.D)+", E="+string(SN.E)+"\quad$"];
        else 
            str=[str;"$\quad\mbox{Flipping }x \rightleftharpoons z\mbox{ axes results in}$";..
        "$D="+string(SN.D)+", E="+string(SN.E)+"\quad$"];
        end
    
    
        SN.Dyy=SS.Dzz;SN.Dzz=SS.Dyy;SN.Dxx=SS.Dxx;
        SN=Dde(SN);
        if 3*abs(SN.E)<abs(SN.D) & badDE then
            str=[str;"$\bf \quad\mbox{Flipping }y \rightleftharpoons z\mbox{ axes results in}$";..
            "$\bf D="+string(SN.D)+", E="+string(SN.E)+"\quad$"];
        else 
            str=[str;"$\quad\mbox{Flipping }y \rightleftharpoons z\mbox{ axes results in}$";..
            "$D="+string(SN.D)+", E="+string(SN.E)+"\quad$"];
        end
    end;
    if use_Stevens==1 then
        str=["$\mbox{The symmetry }C_{1v}$";"$\mbox{leads to the following Hamiltonian:}$"];
        [a,b]=SStev(S,2,0);
        str=[str;"$B_2^0\times$"+b+"$+$"];
        [a,b]=SStev(S,2,1);
        str=[str;"$B_2^1\times[$"+b+"$]+$"];
        [a,b]=SStev(S,2,2);
        str=[str;"$B_2^2\times[$"+b+"$]$"]
        if S.S>1.5 then
           str($)=str($)+"$+$";
           [a,b]=SStev(S,4,0);
           str=[str;"$B_4^0\times[$"+b+"$]+$"];
           [a,b]=SStev(S,4,1);
           str=[str;"$B_4^1\times[$"+b+"$]+$"];
           [a,b]=SStev(S,4,2);
            str=[str;"$B_4^2\times[$"+b+"$]+$"];
           [a,b]=SStev(S,4,3);
            str=[str;"$B_4^3\times[$"+b+"$]+$"];
           [a,b]=SStev(S,4,4);
            str=[str;"$B_4^4\times[$"+b+"$]$"];
        end,
        if S.S>2.5 then
            str($)=str($)+"$+...$";
        end;
    end;
    if use_Stevens==2 then
        str=["$\mbox{The symmetry }C_{2v}$";"$\mbox{leads to the following Hamiltonian:}$"];
        [a,b]=SStev(S,2,0);
        str=[str;"$B_2^0\times $"+b+"$+$"];
        [a,b]=SStev(S,2,2);
        str=[str;"$B_2^2\times[$"+b+"$]$"];
        if S.S>1.5 then
           str($)=str($)+"$+$";
           [a,b]=SStev(S,4,0);
           str=[str;"$B_4^0\times[$"+b+"$]+$"];
           [a,b]=SStev(S,4,2);
            str=[str;"$B_4^2\times[$"+b+"$]+$"];
           [a,b]=SStev(S,4,4);
            str=[str;"$B_4^4\times[$"+b+"$]$"];
        end,
        if S.S>2.5 then
            str($)=str($)+"$+$";
           [a,b]=SStev(S,6,0);
           str=[str;"$B_6^0\times[$"+b+"$]+$"];
           [a,b]=SStev(S,6,2);
            str=[str;"$B_6^2\times[$"+b+"$]+$"];
           [a,b]=SStev(S,6,4);
            str=[str;"$B_6^4\times[$"+b+"$]$"];
           [a,b]=SStev(S,6,6);
            str=[str;"$B_6^6\times[$"+b+"$]$"];
        end;
        if S.S>3.5 then
            str($)=str($)+"$+...$";
        end;
    end;
    if use_Stevens==3 then
        str=["$\mbox{The symmetry }C_{3v}$";"$\mbox{leads to the following Hamiltonian:}$"];
        [a,b]=SStev(S,2,0);
        str=[str;"$B_2^0\times $"+b]; 
        if S.S>1.5 then
           str($)=str($)+"$+$";
           [a,b]=SStev(S,4,0);
           str=[str;"$B_4^0\times[$"+b+"$]+$"];
           [a,b]=SStev(S,4,3);
            str=[str;"$B_4^3\times[$"+b+"$]$"];
        end,
        if S.S>2.5 then
            str($)=str($)+"$+$";
           [a,b]=SStev(S,6,0);
           str=[str;"$B_6^0\times[$"+b+"$]+$"];
           [a,b]=SStev(S,6,3);
            str=[str;"$B_6^3\times[$"+b+"$]+$"];
           [a,b]=SStev(S,6,6);
            str=[str;"$B_6^6\times[$"+b+"$]$"];
        end;
        if S.S>3.5 then
            str($)=str($)+"$+...$";
        end;
    end;
    if use_Stevens==4 then
        str=["$\mbox{The symmetry }C_{4v}$";"$\mbox{leads to the following Hamiltonian:}$"];
        [a,b]=SStev(S,2,0);
        str=[str;"$B_2^0\times $"+b]; 
        if S.S>1.5 then
           str($)=str($)+"$+$";
           [a,b]=SStev(S,4,0);
           str=[str;"$B_4^0\times[$"+b+"$]+$"];
           [a,b]=SStev(S,4,4);
            str=[str;"$B_4^4\times[$"+b+"$]$"];
        end,
        if S.S>2.5 then
            str($)=str($)+"$+$";
           [a,b]=SStev(S,6,0);
           str=[str;"$B_6^0\times[$"+b+"$]+$"];
           [a,b]=SStev(S,6,4);
            str=[str;"$B_6^4\times[$"+b+"$]$"];
        end;
        if S.S>3.5 then
            str($)=str($)+"$+...$";
        end;
    end;
    if use_Stevens==5 then
        str=["$\mbox{The symmetry }C_{6v}$";"$\mbox{leads to the following Hamiltonian:}$"];
        [a,b]=SStev(S,2,0);
        str=[str;"$B_2^0\times $"+b]; 
        if S.S>1.5 then
           str($)=str($)+"$+$";
           [a,b]=SStev(S,4,0);
           str=[str;"$B_4^0\times[$"+b+"$]$"];
        end,
        if S.S>2.5 then
            str($)=str($)+"$+$";
           [a,b]=SStev(S,6,0);
           str=[str;"$B_6^0\times[$"+b+"$]+$"];
           [a,b]=SStev(S,6,6);
            str=[str;"$B_6^6\times[$"+b+"$]$"];
        end;
        if S.S>3.5 then
            str($)=str($)+"$+...$";
        end;
    end;

    hh.button_flip = uicontrol("parent",ani_gui, "style","pushbutton", ..
    "Position",[65 10 100 25], "String","Flip the axes", ..
    "BackgroundColor",[0.6,0.9,0.9], "fontsize",11, ..
    "Relief","raised",..
    "TooltipString", "Flip the anisotropy axes with respect to the main (magnetic) axes", ..
    "Callback","flip_axes()");
    
    if use_Stevens <> %f & use_Stevens <> 0 then
       hh.button_flip.enable='off';
    end

//   hh.button_change = uicontrol("parent",ani_gui, "style","pushbutton", ..
//    "Position",[200 10 100 25], "String","Enter Dnn", ..
//    "BackgroundColor",[0.6,0.9,0.9], "fontsize",11, ..
//    "Relief","raised",..
//    "TooltipString", "Enter Anisotropy parameters directly (see PRL111,127203)", ..
//    "Callback","enter_dnn()");

    hh.button_symmetry = uicontrol("parent",ani_gui, "Position", [220 10 100 25],..
    "Style", "popupmenu",..
    "TooltipString", "Select symmetry class",..
    "BackgroundColor",[0.6,0.9,0.9], "fontsize",11, ..
    "Relief","raised",..
    "String", "D, E model|C1v Symmetry|C2v Symmetry|C3v symmetry|C4v symmetry|C6v symmetry",..
    "Callback", "change_symmetry();",..
    "value", string(use_Stevens+1), ..
    "Tag", "change_sym_button");
    

    hh.button_ok = uicontrol("parent",ani_gui, "style","pushbutton", ..
   "Position",[375 10 100 25], "String","OK", ..
   "BackgroundColor",[0.6,0.9,0.6], "fontsize",11, ..
   "Relief","raised",..
   "TooltipString", "Close submenu", ..
   "Callback","delete(ani_gui)");

   for i=1:size(str,1),
       uicontrol("parent",ani_gui, "style","text", ..
       "Position",[00 255-19*i 560 19], "String",str(i), ..
       "HorizontalAlignment","center",..
       "BackgroundColor",[1,1,1],..
       "fontsize",11);
   end
    //titlepage(str);
    
endfunction


function flip_axes()

    global experiment guicheckbox1 guientry1 guislider1 gui h;
    number=gui.sel_spin.Value
    S.D=experiment.atom(number).D;
    S.E=experiment.atom(number).E;
    SS=Dxyz(S);
    flip= messagebox("Which axes you want to flip?", "modal","question",..
    ["x-y","x-z","y-z","none"]);
    select flip
        case 1 then
            SN.Dxx=SS.Dyy;SN.Dyy=SS.Dxx;SN.Dzz=SS.Dzz;
            SN=Dde(SN);
            experiment.atom(number).D=SN.D;
            experiment.atom(number).E=SN.E;
        
            update_gui_values();
            gui_checkanisotropy();
            cleardata(2); //all calculations are invalid

        case 2 then
            SN.Dxx=SS.Dzz;SN.Dzz=SS.Dxx;SN.Dyy=SS.Dyy;
            SN=Dde(SN);
            experiment.atom(number).D=SN.D;
            experiment.atom(number).E=SN.E;
        
            update_gui_values();
            gui_checkanisotropy();
            cleardata(2); //all calculations are invalid

        case 3 then
            SN.Dyy=SS.Dzz;SN.Dzz=SS.Dyy;SN.Dxx=SS.Dxx;
            SN=Dde(SN);    
            experiment.atom(number).D=SN.D;
            experiment.atom(number).E=SN.E;
            
            update_gui_values();
            gui_checkanisotropy();
            cleardata(2); //all calculations are invalid

    end
   
endfunction

function change_symmetry()
    global experiment gui ani_gui main_gui;
//    global experiment guicheckbox1 guientry1 guislider1 gui ani_gui;
    number=gui.sel_spin.Value;
    pop = get("change_sym_button");
    oldVal = experiment.atom(number).use_Stevens;
    newVal = get(pop, "Value")-1;
    if oldVal <> newVal then
        if experiment.allatomsequ then
            experiment.atom(:).use_Stevens=newVal;
            if oldVal==0 then
                //change from D,E to B_x^y
                for i=1:size(experiment.atom,2),
                    experiment.atom(i).Stevens=zeros(3,7);
                    experiment.atom(i)=DE2Stev(experiment.atom(i));
                    if newVal>2 then
                        experiment.atom(i).Stevens(1,3)=0;
                        //obay symmetry
                    end
                end
            end
            if newVal==0 then
                //change from B_x^y to D, E
                for i=1:size(experiment.atom,2),
                    experiment.atom(i)=Stev2DE(experiment.atom(i));
                end
            end
        else
            experiment.atom(number).use_Stevens=newVal;
            if oldVal==0 then
                //change from D,E to B_x^y
                experiment.atom(number).Stevens=zeros(3,7);
                experiment.atom(number)=DE2Stev(experiment.atom(number));
                if newVal>2 then
                    experiment.atom(number).Stevens(1,3)=0;
                        //obay symmetry
                end
            end
            if newVal==0 then
                //change from B_x^y to D, E
                experiment.atom(number)=Stev2DE(experiment.atom(number));
            end
        end
        
        build_main_gui();
        gui_checkanisotropy();
        
    end

endfunction

function gui_exit()
    
    global main_gui ts_gui cm_gui ani_gui PATHi;

    msg=messagebox("Do you really want to exit ?","modal", "question",..
     ["Yes, save program state" "Yes, don''t save" "No"]);
    if msg<3 then
        close(main_gui);
        if exists('ts_gui') then 
            ierr=execstr('close(ts_gui);','errcatch');
        end;
        if exists('cm_gui') then 
            ierr=execstr('close(cm_gui);','errcatch');
        end;
        if exists('ani_gui') then 
            ierr=execstr('close(ani_gui);','errcatch');
        end;
        disp('bye');
    end
    if msg==1 then  //save last program state 
        file=".LAST_STATE.SOD";
        gui_save(PATHi+file,"SOD")

    end
 
endfunction

function gui_help()
    
    //this function opens the help files which aren't written yet.
    
    url="https://help.scilab.org/";
    if getos()=="Windows"
        winopen(url)
    elseif getos()=="Linux"
        host("xdg-open """+url+"""")
    elseif getos()=="Darwin"         // Mac OS
        host("open """+url+"""")
    else
        warning("openURL: unknown OS")
    end
    
endfunction

function fliptip(experiment)

    global experiment

    tip=experiment.atom(1);
    experiment.atom(1)=experiment.atom(2);
    experiment.atom(2)=tip;
    update_gui_values();
    cleardata(2);

endfunction


// initialize gui
build_main_gui();

// load last-state of the program


last_state_file=".LAST_STATE.SOD";
gui_load(PATHi+last_state_file,"SOD")

