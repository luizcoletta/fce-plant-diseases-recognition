function [results, y, S] = runICEDS(nData, folds, nLabObj, hCla, sFileName, NN, sRan)
% *************************************************************************
% runICEDS: iterative C3E-SL based on Active Learning for detecting new
%           classes
%
% Example: [results, y, S] = runICEDS([1], [1,2,3,4,5]*(-1), [137], [1,2,3], 'detNC-ceratocystis_img11-fs9-res1');
%          [results, y, S] = runICEDS([2], [1,2,3,4,5]*(-1), [236], [1,2,3], 'detNC-ceratocystis_img11-fs9-res2');
%          [results, y, S] = runICEDS([3], [1,2,3,4,5]*(-1), [533], [1,2,3], 'detNC-ceratocystis_img11-fs9-res5');
%          [results, y, S] = runICEDS([4], [1,2,3,4,5]*(-1), [814], [1,2,3], 'detNC-ceratocystis_img11-fs9-res10');
%          [results, y, S] = runICEDS([5], [1,2,3,4,5]*(-1), [1308], [1,2,3], 'detNC-ceratocystis_img11-fs9-res20');
%
% Author: Luiz F. S. Coletta (luiz.fersc@gmail.com) - 01/10/17
% Update: Luiz F. S. Coletta - 10/12/19
% *************************************************************************

path_results = '/results/';

% max time for optimizing
timeOpt = 300; %600; %300;

% set datasets
numDatasets = [1,2,3];
if (nargin >= 1)
    if (nData ~= 0)
        numDatasets = nData;
    end
end

% set folds
numFolds = [1,2,3,4,5];
if (nargin >= 2)
    if (folds ~= 0)
        numFolds = folds;
    end
end

% initial size of the training set (for each dataset)
sizeValSet = [50,200,100];
if (nargin >= 3)
    sizeValSet = nLabObj;
end

% set classes to be hidden
% if max value in a set of a dataset = 0 no classes will be hidden in it
             %data1  %data2    %data3
setHCData = [[1,2,3,-1];[1,2,3,4];[1,2,-1,-1]];
if (nargin >= 4)
    setHCData = hCla;
end

% to save a file with results, set to 1 and choose file name
saveFile = 0;
if (nargin >= 5)
    fileName = sFileName;
    saveFile = 1;
end

H = 5; % number of nearest neighbors (if -1 it will use all objects)
if (nargin >= 6)
    H = NN;
end

% to select objects randomly set to 1 (for the active learning process)
selRandom = 0;
if (nargin >= 7)
    selRandom = sRan;
end

% optimization of the C3E (by using a validation set)
% 1: stochastic search
% 2: grid search
% 3: fixed values
typeS = 3;

% number of select objects for labeling (in each consult)
P2 = 5;

% number of consults to the oracle
numConsults = 5; % but can be 5 for COIL3

printResults = 0;

% 1 to use grid-search in the test set (optimal solution - without validation set)
testSet = 0;

% set alpha and number of iterations (for grid-search)
setAlpha = [0:0.01:0.5]';
setNIter = [1:1:25]';

% choose the validation set
valSet = 1;

sRes = struct('Before',[],'F1',[],'F2',[],'F3',[],'F4',[],'F5',[],'F6',[],'F7',[],'F8',[],'F9',[],'F10',[],'NC',[],'M',[],'V',[],'Mi',[],'Ma',[],'Obj',[],'NameData',[],'Info',[],'Time',[]);
sResCla = struct('Before',[],'C1',[],'C2',[],'C3',[],'C4',[],'C5',[],'C6',[],'C7',[],'C8',[],'C9',[],'C10',[],'C11',[],'C12',[],'C13',[],'C14',[],'C15',[],'C16',[],'C17',[],'C18',[],'C19',[],'C20',[],'M',[],'V',[],'Mi',[],'Ma',[],'NameData',[],'Time',[]);
sResCon = struct('EvalEns',[],'EvalC3E',[],'Ini',[],'Cons1',[],'Cons2',[],'Cons3',[],'Cons4',[],'Cons5',[],'Cons6',[],'Cons7',[],'Cons8',[],'Cons9',[],'Cons10',[]);
data = struct('A','F1','B','F2','C','F3','D','F4','E','F5');
dataCla = struct('A','C1','B','C2','C','C3','D','C4','E','C5','F','C6','G','C7','H','C8','I','C9','J','C10','K','C11','L','C12','M','C13','N','C14','O','C15','P','C16','Q','C17','R','C18','S','C19','T','C20');
nData = fieldnames(data);
nDataCla = fieldnames(dataCla);
nDataCon = fieldnames(sResCon);

results = sResCon;

eval1 = [];
eval2 = [];
SumRes = [];

for i = 1:size(numDatasets,2) % ITERATE DATASETS
    
    results.(nDataCon{3}) = [results.(nDataCon{3}); sResCla];
    for m = 1:numConsults
        results.(nDataCon{m+3}) = [results.(nDataCon{m+3}); sResCla];
    end
    
    MBef1 = [];
    MAft1 = [];
    
    setHClass = setHCData(i,(setHCData(i,:)~=-1));
    
    allClasses = 0;
    if (max(setHClass) == 0)
        setHClass = 1;
        allClasses = 1;
    end
    
    for k = 1:size(setHClass,2) % ITERATE HIDDEN CLASSES
        
        hClass  = setHClass(k);
        
        res1 = sRes;
        res2 = sRes;
        
        delete([pwd, path_results, 'SSet*']);
        
        for j = 1:size(numFolds,2) % ITERATE FOLDS
            
            if (allClasses == 1)
                hClass = 0;
            end
            
            % -------------------------------------------------------------
            % Estimating alfa and number of iterations
            % -------------------------------------------------------------
            if (testSet <= 0)
                [alfa, numit] = validation(numFolds(j), numDatasets(i), hClass, -1, typeS, sizeValSet(i), 'Validation1', timeOpt, valSet);
            end
            
            % -------------------------------------------------------------
            % Running C3E-SL with alfa and numit
            % -------------------------------------------------------------
            if (testSet <= 0)                          
                [r1, y, S, labels, nameData] = runSemiSuperLearning(0, numDatasets(i), 0, sizeValSet(i), numFolds(j), alfa, numit, hClass, -1, '0');
                
                accEns = r1.Before;
                accC3E = r1.F1;
                s = size(r1.EvalEns,1);
                eval1 = [eval1; [hClass*ones(s,1), zeros(s,1), r1.EvalEns]];
                eval2 = [eval2; [hClass*ones(s,1), zeros(s,1), r1.EvalC3E]];
            else
                [r1] = runSemiSuperLearning(0, numDatasets(i), 0, sizeValSet(i), numFolds(j), setAlpha, setNIter, hClass, -1, '0');
                                
                accMax = max(max(r1.F1(2:size(r1.F1,1),2:size(r1.F1,2))));
                [rMax, cMax] = find(r1.F1(2:size(r1.F1,1),2:size(r1.F1,2)) == accMax, 1, 'first');
                bestAlpha = setAlpha(rMax);
                bestNI = setNIter(cMax);
                
                [r1, y, S, labels, nameData] = runSemiSuperLearning(0, numDatasets(i), 0, sizeValSet(i), numFolds(j), bestAlpha, bestNI, hClass, -1, '0');
                
                accEns = r1.Before;
                accC3E = r1.F1;
                s = size(r1.EvalEns,1);
                eval1 = [eval1; [hClass*ones(s,1), zeros(s,1), r1.EvalEns]];
                eval2 = [eval2; [hClass*ones(s,1), zeros(s,1), r1.EvalC3E]];
            end
            
            dta = load(['files_', nameData(1,1:size(nameData,2)-5), '/iTrain0.dat']);
            dte = load(['files_', nameData(1,1:size(nameData,2)-5), '/iTest0.dat']);
           
            NInt = [];
            
            res1.Before = [res1.Before, accEns];
            res1.(data.(nData{j})) = accC3E;
            
            for h = 1:numConsults % ITERATE CONSULTS
                
                if (allClasses == 1)
                    hClass = 0;
                end
                
                % -------------------------------------------------------------
                % Computing classification entropy of objects in the target set
                % -------------------------------------------------------------
                e = zeros(size(y,1),1);
                for r = 1:size(y,1)
                    for s = 1:size(y,2)
                        e(r) = e(r) + (y(r,s)*log2(y(r,s)));
                    end
                    e(r) = e(r)*(-1);
                end
                %e = e/size(y,2); % average entropy
                e = (e-min(e))/(max(e)-min(e)); % normalized entropy
                
                % -------------------------------------------------------------
                % Computing densities of the objects in the target set
                % -------------------------------------------------------------
                if (H < 0)
                    H = size(S,1)-1;
                end
                d = zeros(size(S,1),1);
                for r = 1:size(S,1)
                    nearest = sort(S(r,:),'descend');
                    d(r) = mean(nearest(1:H)); % normalized
                end
                %d = ones(size(y,1),1);
                
                % -------------------------------------------------------------
                % Selecting objects to be labeled according to E and D
                % -------------------------------------------------------------
                w = zeros(P2,1);
                g = zeros(size(S,1),1);
                eAux = e;
                for r = 1:P2
                    if (selRandom == 0)
                        selIndex = 1;
                    else
                        selIndex = round((size(S,1)-1)*rand);
                        if (selIndex == 0)
                            selIndex = 1;
                        end     
                    end
                    if (r == 1)
                        [aux1,aux2] = sort(eAux.*d,'descend');
                        w(r) = aux2(selIndex);
                        eAux(aux2(selIndex)) = 0;
                    else
                        for s = 1:size(S,1)
                            for t = 1:r-1
                                g(s) = 1-((1/(r-1))*S(s,w(t))); % normalized
                            end
                        end
                        [aux1,aux2] = sort((eAux.*d).*g,'descend');
                        w(r) = aux2(selIndex);
                        eAux(aux2(selIndex)) = 0;
                    end
                    SumRes = [SumRes; [i, hClass, j, h, e(aux2(selIndex)), d(aux2(selIndex)), g(aux2(selIndex)), aux2(selIndex)]];
                end
                w = sort(w','descend');
             
                NewInstances = labels(w);
                
                dlmwrite(['files_', nameData(1,1:size(nameData,2)-5), '/selObjValues1.dat'], SumRes, 'delimiter', '\t');
                dlmwrite(['files_', nameData(1,1:size(nameData,2)-5), '/selObjValues2.dat'], [dte(w,:),w'], '-append', 'delimiter', '\t');
                
                % Adjusting indexes of the new labeled objects
                % -------------------------------------------------------------
                x1 = zeros(1,size(S,1)+size(NInt,2));
                x1(:,NInt) = -1;
                
                x2 = ones(1,size(S,1));
                x2(:,w) = -2;
                
                x3 = x1;
                x3(:,(x3~=(-1))) = x2;
                x4 = find(x3 < 0);
                
                NInt = sort(x4,'descend');
                % -------------------------------------------------------------
                
                % -------------------------------------------------------------
                % Estimating alfa and number of iterations
                % -------------------------------------------------------------
                if (testSet <= 0)
                    [alfa, numit] = validation(numFolds(j), numDatasets(i), hClass, NInt, typeS, sizeValSet(i), 'Validation2', timeOpt, valSet);
                end
                
                % -------------------------------------------------------------
                % Running C3E-SL with alfa and numit
                % -------------------------------------------------------------
                if (testSet <= 0)
                    [r2, y, S, labels, nameData] = runSemiSuperLearning(0, numDatasets(i), 0, sizeValSet(i), numFolds(j), alfa, numit, hClass, NInt, num2str(h));
                    
                    accEns = r2.Before;
                    accC3E = r2.F1;
                    s = size(r2.EvalEns,1);
                    eval1 = [eval1; [hClass*ones(s,1), h*ones(s,1), r2.EvalEns]];
                    eval2 = [eval2; [hClass*ones(s,1), h*ones(s,1), r2.EvalC3E]];
                else
                    [r2] = runSemiSuperLearning(0, numDatasets(i), 0, sizeValSet(i), numFolds(j), setAlpha, setNIter, hClass, NInt, num2str(h));
                    
                    accMax = max(max(r2.F1(2:size(r2.F1,1),2:size(r2.F1,2))));
                    [rMax, cMax] = find(r2.F1(2:size(r2.F1,1),2:size(r2.F1,2)) == accMax, 1, 'first');
                    bestAlpha = setAlpha(rMax);
                    bestNI = setNIter(cMax);
                    
                    [r2, y, S, labels, nameData] = runSemiSuperLearning(0, numDatasets(i), 0, sizeValSet(i), numFolds(j), bestAlpha, bestNI, hClass, NInt, num2str(h));
                    
                    accEns = r2.Before;
                    accC3E = r2.F1;
                    s = size(r2.EvalEns,1);
                    eval1 = [eval1; [hClass*ones(s,1), h*ones(s,1), r2.EvalEns]];
                    eval2 = [eval2; [hClass*ones(s,1), h*ones(s,1), r2.EvalC3E]];
                end
                
                if (allClasses == 1)
                    hClass = 1;
                end
                
                res2.Info = r2.Info;
                res2.NameData = r2.NameData;
                res2.(data.(nData{j})) = accC3E;
                if (j>1)
                    res2.Before = [results.(nDataCon{h+3})(i).(dataCla.(nDataCla{hClass})).Before, accEns];
                    res2.NC = [results.(nDataCon{h+3})(i).(dataCla.(nDataCla{hClass})).NC, NewInstances];
                else
                    res2.Before = accEns;
                    res2.NC = NewInstances;
                end
                
                results.(nDataCon{h+3})(i).(dataCla.(nDataCla{hClass})) = res2;
                
                mFold = 0;
                for m = 1:j
                    rAux = results.(nDataCon{h+3})(i).(dataCla.(nDataCla{hClass})).(data.(nData{m}));
                    mFold = mFold + rAux;
                end
                results.(nDataCon{h+3})(i).(dataCla.(nDataCla{hClass})).M = mFold/j;
                
                if (saveFile > 0)
                    save([pwd, path_results, fileName, '.mat'], 'results');
                end
                
                if (printResults==1)
                    print(NInt, dta, dte, [pwd, path_results, strrep(res2.NameData, '.arff', ''), num2str(hClass), num2str(j), num2str(h)], hClass, j, i, nameData, selRandom);
                end
                
                %%%% AQUI ACESSA DADOS DE TREINO E TESTE
                dta = load(['files_', nameData(1,1:size(nameData,2)-5), '/iTrain', num2str(h), '.dat']);
                dte = load(['files_', nameData(1,1:size(nameData,2)-5), '/iTest', num2str(h), '.dat']);
                
            end % for consults
            
        end % for folds
        
        for n = 1:numConsults
            results.(nDataCon{n+3})(i).Before = [results.(nDataCon{n+3})(i).Before, mean(results.(nDataCon{n+3})(i).(dataCla.(nDataCla{hClass})).Before)];
            results.(nDataCon{n+3})(i).M = [results.(nDataCon{n+3})(i).M, results.(nDataCon{n+3})(i).(dataCla.(nDataCla{hClass})).M(2,2)];
            results.(nDataCon{n+3})(i).NameData = results.(nDataCon{n+3})(i).(dataCla.(nDataCla{hClass})).NameData;
        end
        
        res1.Time = 0;
        mFold = 0;
        for m = 1:size(folds,2)
            rAux = res1.(data.(nData{m}));
            mFold = mFold + rAux(2,2);
        end
        res1.M(2,2) = mFold/size(folds,2);
        res1.NameData = r1.NameData;
        res1.Info = r1.Info;
        
        results.(nDataCon{3})(i).(dataCla.(nDataCla{hClass})) = res1;
        
        MBef1 = [MBef1, mean(res1.Before)];
        MAft1 = [MAft1, res1.M(2,2)];
    end % for hidden classes
    
    results.(nDataCon{1}) = eval1;
    results.(nDataCon{2}) = eval2;
    
    results.(nDataCon{3})(i).Before = MBef1;
    results.(nDataCon{3})(i).M = MAft1;
    results.(nDataCon{3})(i).NameData = r1.NameData;
    
    if (saveFile > 0)
        save([pwd, path_results, fileName, '.mat'], 'results');
    end
end

end

% -------------------------------------------------------------
% Optimizing C3E parameters
% -------------------------------------------------------------
function [alfa, numit] = validation(fold, data, mc, nit, typeSearch, sizeValSet, nameFile, time, valSet)

alfa = 0.001;
numit = 1;

% Stochastic Search (D2E)
if (typeSearch == 1)
    best = searching(fold, data, valSet, mc, nit, sizeValSet, nameFile, time);
    alfa = best(1);
    numit = best(2);
end

% Grid Search
if (typeSearch == 2)
    
    alphaRange = [0:0.01:0.5]';
    numitRange = [1:1:25]';
    
    rR1 = runSemiSuperLearning(0, data, 1, sizeValSet, fold, alphaRange, numitRange, mc, nit, '0');
    rR2 = runSemiSuperLearning(0, data, 2, sizeValSet, fold, alphaRange, numitRange, mc, nit, '0');
    
    al = rR1.F1(2:size(rR1.F1,1),1);
    ni = rR1.F1(1,2:size(rR1.F1,2));
    [a,b] = find(rR1.F1(2:size(rR1.F1,1),2:size(rR1.F1,2)) == max(max(rR1.F1(2:size(rR1.F1,1),2:size(rR1.F1,2)))));
    c1 = [al(a),ni(b)'];
    [a,b] = find(rR2.F1(2:size(rR1.F1,1),2:size(rR1.F1,2)) == max(max(rR2.F1(2:size(rR1.F1,1),2:size(rR1.F1,2)))));
    c2 = [al(a),ni(b)'];
    mc = mean([c1;c2]);
    mc(2) = round(mc(2));
    alfa = mc(1);
    numit = mc(2);
end
end

% -------------------------------------------------------------
% Stochastic Search
% -------------------------------------------------------------
function [best] = searching(fold, data, val, mc, nit, sValSet, nameFile, time)

% http://www1.icsi.berkeley.edu/~storn/code.html (Rainer Storn)
opar = [fold,data,val,mc,sValSet,nit];
VTR = -1;         % VTR "Value To Reach" (stop when ofunc < VTR)
D = 2;            % D number of parameters of the objective function
XVmin = [0 1];
XVmax = [0.15 10]; % y problem data vector (remains fixed during optimization)
itermax = 10000;   % itermax maximum number of iterations (generations)
st = tic;

% strategy       1 --> DE/best/1/exp           6 --> DE/best/1/bin
%                2 --> DE/rand/1/exp           7 --> DE/rand/1/bin
%                3 --> DE/rand-to-best/1/exp   8 --> DE/rand-to-best/1/bin
%                4 --> DE/best/2/exp           9 --> DE/best/2/bin
%                5 --> DE/rand/2/exp           else  DE/rand/2/bin
strategy = 9;

NP = 20;     % NP number of population members
vF = 0.25;   % F DE-stepsize F ex [0, 2]
vCR = 0.25;  % CR crossover probabililty constant ex [0, 1]

[best,f,~,~,~,~] = optD2E(opar,'ofD2E',VTR,D,XVmin,XVmax,NP,itermax,strategy,st,time,nameFile,1,vF,vCR);
best = [best, (100-f)];
end

% -------------------------------------------------------------
% Print results
% -------------------------------------------------------------
function [] = print(w, dta, dte, name, class, fold, dataset, nameData, selRandom)

if (selRandom == 0)
    ni = load(['files_', nameData(1,1:size(nameData,2)-5), '/iNewInst1.dat']);
else
    ni = [dte(w',1),dte(w',2), dte(w',3)];
end

figure('Color','white', 'Visible', 'off');
hold on;

cores = [0,0,1; 0,0.6,0; 1,0,0; 0.8,0.4,0; 0,1,0; 0.6,0,0.6; 0.7,0.6,0; 0,0.2,0.6; 0,0.7,0.5];
marker = ['s';'o';'^';'v';'+';'.';'x';'*';'p';'h';'>';'<'];

% GREEN (TEST)
for g = 1:max(dte(:,3))
    scatter(dte((dte(:,3)==g), 1), dte((dte(:,3)==g), 2), 'marker', marker(g,:), 'markerfacecolor', cores(2,:), 'markeredgecolor', cores(2,:));
end

% BLUE (TRAIN)
for g = 1:max(dta(:,3))
    scatter(dta((dta(:,3)==g), 1), dta((dta(:,3)==g), 2), 'marker', marker(g,:), 'markerfacecolor', cores(1,:), 'markeredgecolor', cores(1,:));
end

% RED (NEW INSTANCES)
for g = 1:max(ni(:,3))
    scatter(ni((ni(:,3)==g), 1), ni((ni(:,3)==g), 2), 'marker', marker(g,:), 'markerfacecolor', cores(3,:), 'markeredgecolor', cores(3,:));
end

saveas(gcf, [name,'.eps'], 'epsc');
saveas(gcf, [name,'.png'], 'png');
hold off;
end
