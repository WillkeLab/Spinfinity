classdef dIdVexperiment
    % dIdVexperiment - A class to represent a dI/dV experiment with predefined parameters and values.
    %HIJACKS MARKUS TERNES CODE AND ALLOWS TO RUN IT IN MATLAB!!

    %In order to not click enter when Spin-Gui is called you need to change
    %the line 4028: answ=x_dialog('Script for execution',scrpt); to
    %answ=scrpt;
    properties
        T = 0.0861733
        xrange = 10
        lt = 0.005
        ptip = [0,0,0]
        psample = [0,0,0]
        ptipK = []
        psampleK = []
        atom = struct('S', 0.5, 'g', 2, 'D', 0, 'E', 0, 'alpha', 0, 'beta', 0, 'gamma', 0, 'J', -0.04, 'U', 0, 'w', 20)
        Eigenvec = []
        Eigenval = []
        position = 1
        jposition = 1
        A = 1
        T_ts = [1,1,1]
        b = 0
        x0 = 0
        y0 = 0
        B = [0,0,4]
        matrix = '"1"'
        matrixDM = '"0, 0, 0"'
        Phase = '"0"'
        heisenberg_coupling = 0
        sef = 1
        paramagnetic = 0
        
        paramag_S = 2.5
        paramag_g = 2
        eta = 0.3
        no_eval = 1000
        spec_length = 1000
        % max_no_eigenstates = 50 % Uncomment if needed
        third_order_calc = 0
        rate_calc = 0
        entanglement = 0
        allatomsequ = 0
        func_tip = 0
        corr_sample = 0
        corr_tip = 0
        spectrum_deriv = 1
        use_T_ts = 0
        RedF_coeff = []
        RedF_dm_init = []
        savedata = 1
        Bias = [];
        dIdV = [];
        parameter2 = [];
        parameter2Label = [];
        offset = 0;
    end

    methods
        function obj = dIdVexperiment(nAtoms)
            % Constructor method for creating an instance of the class
            % with default properties initialized as per the provided structure.
            for i = 1:nAtoms
                obj.atom(i) = struct('S', 0.5, 'g', 2, 'D', 0, 'E', 0, 'alpha', 0, 'beta', 0, 'gamma', 0, 'J', -0.04, 'U', 0, 'w', 20);
                obj.position = nAtoms;
                obj.jposition = nAtoms;
            end
        end

        function [Bias, dIdV] = execSpinGui(obj)
            % Define the lines of text as a cell array
            linesOfCode = {
                '%savedata=%T'
                'draw_spectrum();'
                'quit();'
                };

            scriptFilename = "script.txt";  % Specify the filename
            fileID = fopen(scriptFilename, 'w'); % Open the file with write permission
            % Write each line of text to the file
            for i = 1:length(linesOfCode)
                fprintf(fileID, '%s\n', linesOfCode{i});
            end
            fclose(fileID); % Close the file

            obj.exportExperimentParameters(scriptFilename);

            try
                delete data_inelastic_all1.dat
            catch
                disp('File not found or could not be deleted.')
            end

            % Define the Scilab commands as strings
            command1 = 'exec("Spin-Gui-1.9.5_mod.sce")';
            command2 = 'gui.nbr_spins.Value = 2; number_spins_callback(); '; %quit()
            command3 = 'gui_script(mgetl(''export.txt''))';


            % Combine the commands into a single string to be executed in Scilab
            combinedCommands = sprintf('%s\n%s\n%s', command1, command2, command3);

            % Create a temporary script file
            tempScriptFilename = 'tempScilabScript.sce';
            fid = fopen(tempScriptFilename, 'wt');
            fprintf(fid, '%s', combinedCommands);
            fclose(fid);
            scilabEXE = """C:\Program Files\scilab-2024.1.0\bin\WScilex.exe""";

            % Call Scilab with the temporary script
            [status, result] = system(strjoin([scilabEXE ' -f ""tempScilabScript.sce""']));

            % Optionally, delete the temporary script file after execution
            delete(tempScriptFilename);

            % Display the output
            disp(result);

            % Read the data file
            if exist('data_inelastic_all1.dat', 'file')
                datainelasticall1 = readtable("data_inelastic_all1.dat");
                Bias = datainelasticall1.Var1;
                dIdV = datainelasticall1.Var2;
            else
                error('Data file not found.');
            end

            if(size(obj.atom,2)>1)
                datainelasticall2 = readtable("data_inelastic_all2.dat");
                Bias = datainelasticall2.Var1;
                dIdV(:,2) = datainelasticall2.Var2;
                dIdV(:,1) = datainelasticall1.Var2;
            end
                        if(size(obj.atom,2)>2)
                datainelasticall3 = readtable("data_inelastic_all3.dat");
                Bias = datainelasticall3.Var1;
                dIdV(:,3) = datainelasticall3.Var2;
            end

            delete script.txt export.txt *.dat
        end


        function plotdIdV(obj)
            figure(123); hold on
            plot(obj.Bias,obj.dIdV+obj.offset,"LineWidth",2)
            xlabel("Voltage (mV)"); ylabel("dIdV (a.u.)")
        end
        function plotdIdVAll(obj)
            figure(126); hold on
            for i = 1:size(obj.dIdV,2)
                plot(obj.Bias,obj.dIdV(:,i)+obj.offset,"LineWidth",2)
            end
            xlabel("Voltage (mV)"); ylabel("dIdV (a.u.)")
        end
        function plot2Ddata(obj,Bias,parameter2,dIdV,parameter2Label)
            figure(124); hold on
            imagesc(Bias,parameter2,dIdV)
            xlabel("Voltage (mV)"); ylabel(parameter2Label);
            cb = colorbar; colormap("jet");
            xlim([min(Bias) max(Bias)]); ylim([min(parameter2) max(parameter2)]);
            lbl = ylabel(cb, 'Intensity');
            set(lbl, 'FontSize', 12, 'Rotation', 270);
        end
    end
    methods (Access = private)
        function exportExperimentParameters(obj, scriptFilename)
            % Function to export all parameters of an 'experiment' structure to a text file
            % and append the content of an external text file specified by 'scriptFilename'.

            % Open the file with write permission
            fileId = fopen('export.txt', 'w');

            % Check if the file is opened successfully
            if fileId == -1
                error('Failed to open the file for writing.');
            end

            % Define a recursive function to print each field
            function printFields(prefix, structData)
                fields = fieldnames(structData);
                for i = 1:numel(fields)
                    fieldName = fields{i};
                    fieldValue = structData.(fieldName);
                    fullFieldName = [prefix, fieldName];
                    if isstruct(fieldValue)
                        % If the field is a nested structure, call recursively
                        printFields([fullFieldName, '.'], fieldValue);
                    else
                        % Convert the field value to a string representation
                        if isnumeric(fieldValue) || islogical(fieldValue)
                            valueStr = mat2str(fieldValue);
                        elseif ischar(fieldValue)
                            valueStr = fieldValue;
                        else
                            valueStr = 'Non-printable value';
                        end
                        % Print the field name and value to the file
                        fprintf(fileId, 'experiment.%s = %s;\n', fullFieldName, valueStr);

                    end
                end
                if(size(obj.atom,2)>1)
                    for iatom = 1:size(obj.atom,2)
                        listofAtomProp = {'S','g','D','E','alpha','beta','gamma','J','U','w'}
                        for j = 1:length(listofAtomProp)
                            fprintf(fileId, 'experiment.atom(%s).%s = %s;\n', string(iatom), listofAtomProp{j}, string(obj.atom(iatom).(listofAtomProp{j})))
                        end
                    end
                end
            end

            % Start the recursive printing of fields
            printFields('', obj);

            % Check if 'scriptFilename' is provided and not empty
            if nargin > 1 && ~isempty(scriptFilename)
                % Check if the script file exists
                if exist(scriptFilename, 'file') == 2
                    % Open the script file for reading
                    scriptFileId = fopen(scriptFilename, 'r');
                    % Check if the script file is opened successfully
                    if scriptFileId == -1
                        error('Failed to open the script file for reading.');
                    end
                    % Read the content of the script file and write it to the export file
                    while ~feof(scriptFileId)
                        line = fgets(scriptFileId);
                        fprintf(fileId, '%s', line);
                    end
                    % Close the script file
                    fclose(scriptFileId);
                else
                    error('The script file does not exist.');
                end
            end

            % Close the export file
            fclose(fileId);
        end
    end
end
