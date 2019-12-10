function [tableHist, tableAcc, performance, performanceSD, tableAccAll, tableOptAlpha, tableOptI, tFreqAlpha, tFreqI, tBalAcc] = getResults(results, data, hClasses, maxClasses, nConsults, genPlots)
% *************************************************************************
% getResAL: function to get results in the file 'results.mat' provided by the
%           IC-EDS. We consider 5-fold cross-validation.
%           
% Example: [tableHist, tableAcc, performance, performanceSD] = getResults(results, 1, [1,2,3], 3, 5, 0); 
%
% Author: Luiz F. S. Coletta (luiz.fersc@gmail.com) - 06/02/14
% Update: Luiz F. S. Coletta - 10/12/19
% *************************************************************************

tableHist = [];
tableAcc = [];
tableAccAll = [];
tableOptAlpha = [];
tableOptI = [];

for k = 1:size(hClasses,2) % k Classes
    for j = data:data      % j Dataset
        
        class = hClasses(k);
        dataset = j;
        
        sResCon = struct('EvalEns',[],'EvalC3E',[],'Ini',[],'Cons1',[],'Cons2',[],'Cons3',[],'Cons4',[],'Cons5',[],'Cons6',[],'Cons7',[],'Cons8',[],'Cons9',[],'Cons10',[]);
        nDataCon = fieldnames(sResCon);
        
        sResCla = struct('Before',[],'C1',[],'C2',[],'C3',[],'C4',[],'C5',[],'C6',[],'C7',[],'C8',[],'C9',[],'C10',[],'C11',[],'C12',[],'C13',[],'C14',[],'C15',[],'C16',[],'C17',[],'C18',[],'C19',[],'C20',[],'M',[],'V',[],'Mi',[],'Ma',[],'NameData',[],'Time',[]);
        nDataClass = fieldnames(sResCla);
        
        tbAcc = zeros(nConsults+1,2);
        
        name = strrep(results.Ini(dataset).NameData, '.arff', ''); % name dataset
        
        ACons = [];
        ICons = [];
        
        for i = 1:nConsults+1 % i Consults
            
            ens = roundn(mean(results.(nDataCon{i+2})(dataset).Before(k)),-2);
            c3e = roundn(mean(results.(nDataCon{i+2})(dataset).M(k)),-2);
            tbAcc(i,:) = [ens,c3e];
            tableAcc = [tableAcc; [class,ens,c3e]];
            
            if (class == hClasses(1)) % considering all classes
                ensAll = roundn(mean(results.(nDataCon{i+2})(dataset).Before),-2);
                c3eAll = roundn(mean(results.(nDataCon{i+2})(dataset).M),-2);
                tableAccAll = [tableAccAll; [i,ensAll,c3eAll]];
            end
            
            % GETTING DE OPTIMAL PARAMETERS VALUES
            a1 = results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).F1(2,1);
            a2 = results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).F2(2,1);
            
            % COMENTAR PARA MENOS FOLDS
            a3 = results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).F3(2,1);
            a4 = results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).F4(2,1);
            a5 = results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).F5(2,1);
            
            i1 = results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).F1(1,2);
            i2 = results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).F2(1,2);
            
            % COMENTAR PARA MENOS FOLDS
            i3 = results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).F3(1,2);
            i4 = results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).F4(1,2);
            i5 = results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).F5(1,2);
            
            % COMENTAR PARA MENOS FOLDS
            ACons = [ACons; [a1; a2; a3; a4; a5]];
            ICons = [ICons; [i1; i2; i3; i4; i5]];
            
            %ACons = [ACons; [a1; a2]];
            %ICons = [ICons; [i1; i2]];
            
            if i > 1 % histogram
                
                elem = numel(results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).NC);
                
                % HISTOGRAM PLOTS
                if (genPlots == 1)
                    fig = figure('Color','white', 'Visible', 'off');
                    hist(reshape(results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).NC,1,elem),(1:maxClasses));
                    set(gca,'LooseInset',get(gca,'TightInset'));
                    saveas(fig, ['hist-',name,'-class',num2str(class),'-cons',num2str(i-1),'.png'], 'png');
                    % to save in pdf
                    %set(fig,'Units','Inches');
                    %pos = get(fig,'Position');
                    %set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
                    %saveas(fig, ['hist-',name,'-class',num2str(class),'-cons',num2str(i-1),'.pdf'], 'pdf');
                end
                
                h = hist(reshape(results.(nDataCon{i+2})(dataset).(nDataClass{class+1}).NC,1,elem),(1:maxClasses));
                tableHist = [tableHist; [class,h,(h/elem*100)]];
            end
        end
        
        tableOptAlpha = [tableOptAlpha, ACons];
        tableOptI = [tableOptI, ICons];
        
        % ACCURACIES PLOTS (PER CLASS)
        if (genPlots == 1)
            fig = figure('Color','white', 'Visible', 'off');
            %bar(tbAcc,'DisplayName',results.Ini(dataset).NameData)
            plot(tbAcc(:,1),'LineStyle','--','LineWidth',1,'Marker','o','Color','b','MarkerSize',8); hold on;
            plot(tbAcc(:,2),'LineStyle','-.','LineWidth',1,'Marker','x','Color','r','MarkerSize',8);
            set(gca,'XTick',(1:nConsults+1));
            set(gca,'FontSize',13, 'FontName', 'Helvetica');
            grid on;
            box off;
            set(gca,'LooseInset',get(gca,'TightInset'));
            %ylim([50 100]);
            %set(gca, 'XMinorTick', 'off', 'YMinorTick', 'off');
            %title_handle = title(results.Ini(dataset).NameData);
            saveas(fig, [name,'-class',num2str(class),'.png'], 'png');
            % to save in pdf
            %set(fig,'Units','Inches');
            %pos = get(fig,'Position');
            %set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
            %saveas(fig, [name,'-class',num2str(class),'.pdf'], 'pdf');
        end
        
        % ACCURACIES PLOTS (AVERAGE ALL CLASSES)
        if (genPlots == 1)
            if (class == hClasses(1))
                fig = figure('Color','white', 'Visible', 'off');
                %bar(tbAcc,'DisplayName',results.Ini(dataset).NameData)
                plot(tableAccAll(:,2),'LineStyle','--','LineWidth',1,'Marker','o','Color','b','MarkerSize',8); hold on;
                plot(tableAccAll(:,3),'LineStyle','-.','LineWidth',1,'Marker','x','Color','r','MarkerSize',8);
                set(gca,'XTick',(1:nConsults+1));
                set(gca,'FontSize',13, 'FontName', 'Helvetica');
                grid on;
                box off;
                set(gca,'LooseInset',get(gca,'TightInset'));
                %ylim([50 100]);
                %set(gca, 'XMinorTick', 'off', 'YMinorTick', 'off');
                %title_handle = title(results.Ini(dataset).NameData);
                saveas(fig, [name,'-AllClasses.png'], 'png');
                % to save in pdf
                %set(fig,'Units','Inches');
                %pos = get(fig,'Position');
                %set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
                %saveas(fig, [name,'-class',num2str(class),'.pdf'], 'pdf');
            end
        end
    end
end

% COMENTAR PARA MENOS FOLDS
sOpt = size(tableOptAlpha,2);

aux = tableOptAlpha;
tableOptAlFold = [aux(1:5,1:sOpt), aux(6:10,1:sOpt), aux(11:15,1:sOpt), aux(16:20,1:sOpt), aux(21:25,1:sOpt), aux(26:30,1:sOpt)];
tableOptAlFold = tableOptAlFold';

values = reshape(tableOptAlFold,numel(tableOptAlFold),1);
u = unique(values);
tFreqAlpha = [u histc(values,u)];

tableOptAlFold = [tableOptAlFold; std(tableOptAlFold); max(tableOptAlFold); min(tableOptAlFold); mean(tableOptAlFold)];

aux = tableOptI;
tableOptIFold = [aux(1:5,1:sOpt), aux(6:10,1:sOpt), aux(11:15,1:sOpt), aux(16:20,1:sOpt), aux(21:25,1:sOpt), aux(26:30,1:sOpt)];
tableOptIFold = tableOptIFold';

values = reshape(tableOptIFold,numel(tableOptIFold),1);
u = unique(values);
tFreqI = [u histc(values,u)];

tableOptIFold = [tableOptIFold; std(tableOptIFold); max(tableOptIFold); min(tableOptIFold); mean(tableOptIFold)];

tableOptAlpha = [tableOptAlpha; std(tableOptAlpha); max(tableOptAlpha); min(tableOptAlpha); mean(tableOptAlpha)];
tableOptI = [tableOptI; std(tableOptI); max(tableOptI); min(tableOptI); mean(tableOptI)];

tableOptAlpha = 0;
tableOptI = 0;
tFreqAlpha = 0;
tFreqI = 0;


%--------------------------------------------------------------------------
indSum = results.EvalEns(:,4)==0;
balAccEns = [results.EvalEns(indSum,1:3),results.EvalEns(indSum,6)];
balAccC3E = [results.EvalC3E(indSum,1:3),results.EvalC3E(indSum,6)];
tBalAcc = [];

for i = 1:size(hClasses,2) % i = Classes
    
    indClass = balAccEns(:,1)==i;
    balAccClassEns = balAccEns(indClass,:);
    balAccClassC3E = balAccC3E(indClass,:);
    
    for j = 0:nConsults % j = Consults
    
        indtBalAcc = balAccClassEns(:,2)==j;
        tBalAcc = [tBalAcc;[j,mean(balAccClassEns(indtBalAcc,4))*100,mean(balAccClassC3E(indtBalAcc,4))*100]];
        
    end 
end    
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
performance = [];
% COMENTAR PARA MENOS FOLDS
for i = 1:size(hClasses,2) % i = Classes
    
    cLOENS1 = results.EvalEns(results.EvalEns(:,1)==i,1:12);
    cLOENS2 = cLOENS1(cLOENS1(:,4)==i,1:12);
    
    for j = 0:nConsults % j = Consults
        
        cLOENS3 = cLOENS2(cLOENS2(:,2)==j,1:12);
        performance = [performance;[i,j,mean(cLOENS3(:,6:12))]];
    end 
end    
for i = 1:size(hClasses,2) % i = Classes
    
    cLOENS1 = results.EvalC3E(results.EvalC3E(:,1)==i,1:12);
    cLOENS2 = cLOENS1(cLOENS1(:,4)==i,1:12);
    
    for j = 0:nConsults % j = Consults
        
        cLOENS3 = cLOENS2(cLOENS2(:,2)==j,1:12);
        performance = [performance;[i,j,mean(cLOENS3(:,6:12))]];
    end 
end    
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
performanceSD = [];
% COMENTAR PARA MENOS FOLDS
for i = 1:size(hClasses,2) % i = Classes
    
    cLOENS1 = results.EvalEns(results.EvalEns(:,1)==i,1:12);
    cLOENS2 = cLOENS1(cLOENS1(:,4)==i,1:12);
    
    for j = 0:nConsults % j = Consults
        
        cLOENS3 = cLOENS2(cLOENS2(:,2)==j,1:12);
        performanceSD = [performanceSD;[i,j,std(cLOENS3(:,6:12))]];
    end 
end    
for i = 1:size(hClasses,2) % i = Classes
    
    cLOENS1 = results.EvalC3E(results.EvalC3E(:,1)==i,1:12);
    cLOENS2 = cLOENS1(cLOENS1(:,4)==i,1:12);
    
    for j = 0:nConsults % j = Consults
        
        cLOENS3 = cLOENS2(cLOENS2(:,2)==j,1:12);
        performanceSD = [performanceSD;[i,j,std(cLOENS3(:,6:12))]];
    end 
end    
%--------------------------------------------------------------------------

clear class dataset sResCon nDataClass ens c3e elem h name data genPlots i j k hClasses maxClasses nConsults nDataCon sResCla tbAcc
clear a1 a2 a3 a4 a5 i1 i2 i3 i4 i5 ACons ICons aux values u cLOENS1 cLOENS2 cLOENS3
