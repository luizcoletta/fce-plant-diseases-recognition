function [results, y, SSet, labels, nameData] = runSemiSuperLearning(savefl, nData, val, nObjFold, folds, a, n, missClass, newObj, infoIter)
% *************************************************************************
% runSemiSuperLearning: run ensemble (codes in java) and C3E-SL (codes in
%                       matlab) varying alpha and the number of iterations
%                       assuming a certain number of labeled objects (as a 
%                       training set), the rest will be the test set. 
%                       Determining the number of folds, this setting will 
%                       be repeated for exclusive combinations of objects 
%                       in the training set (the seeder always allows to 
%                       generate the same sets).              
%              
% Example:  [results, y, SSet, labels] = runSemiSuperLearning(0, [3], 0, 50, 3, [0:0.1:0.5]', [1:1:10]');
%           
% Author: Luiz F. S. Coletta (luiz.fersc@gmail.com) - 08/03/12
% Update: Luiz F. S. Coletta - 28/10/17
% *************************************************************************

data_path = '/data/';

% set 1 for saving results
saveFile = 0;
if (nargin >= 1)
    saveFile = savefl;
end

% numeric vector setting databases
numDatasets = [1, 2, 3];
if (nargin >= 2)
    if (nData ~= 0)
        numDatasets = nData;
    end
end

setOfData = 3; % choose the set of databases
if (setOfData == 1) 
    data = struct('A','coil3.arff','B','iris.arff');
end
if (setOfData == 2) 
    data = struct('A','caltech6vgg16.arff','B','corel1000vgg16.arff','C','producevgg16.arff','D','coil20vgg16.arff', 'F','ceratocystis.arff');
end
if (setOfData == 3) 
    data = struct('A','ceratocystis1.arff','B','ceratocystis2.arff','C','ceratocystis5.arff','D','ceratocystis10.arff','E','ceratocystis20.arff');
end

% 0: for testing (to build the test and training sets);
% 1: for validation (fold 1 of 2 from labeled objects - the dataset's name appears with "R");
% 2: for validation (fold 2 of 2 from labeled objects - the dataset's name appears with "R");
validation = 0;
if (nargin >= 3)
    validation = val;
end

% number of objects in each fold
numObjFold = 0;
if (nargin >= 4)
    if (nObjFold(1) > 0)
        numObjFold = nObjFold;
    end
end

% number of folds. To use all data, totalFolds has to be the total number 
% of objects divided by numObjFold
totalFolds = 0;
if (nargin >= 5) 
   totalFolds = folds;
end

% set alpha's vector
alpha = [0:0.001:1]';
if (nargin >= 6)
    alpha = a;
end

% set number of iterations
numIter = [1:1:50]';
if (nargin >= 7)
    numIter = n;
end

%  0: the train set will contain all classes
% >0: the train set will not contain the class of this index
mc = 0;
strmc = '0';
if (nargin >= 8)
    mc = missClass;
    strmc = [];
    for i = 1:size(mc,2)
        if (mc(i) == 0)
            strmc = '0';
            break;
        end
        strmc = [strmc,num2str(mc(i))];
    end
end

%  <0: there are no new objects
% >=0: there are new objects (from indexes numbers)
nInst = -1;
strni = '0';
if (nargin >= 9)
    nInst = newObj;
    if (nInst(1) > 0)
        strni = '1';
    end
end

% number of iteration of the algorithm for identifying new classes
iter = '0';
if (nargin >= 10)
    iter = num2str(infoIter);
end

% C3E version (0: Original C3E; 1: Fast C3E-SL; 2: C3E-SL)  
C3EVersion = 1;

nData = fieldnames(data);
results = [];

s = struct('Before',[],'F1',[],'F2',[],'F3',[],'F4',[],'F5',[],'F6',[],'F7',[],'F8',[],'F9',[],'F10',[],'M',[],'V',[],'Mi',[],'Ma',[],'Obj',[],'NameData',[],'Info',[],'Time',[],'EvalEns',[],'EvalC3E',[]);

for i = 1:size(numDatasets,2)
    
    nameData = data.(nData{numDatasets(i)});
    aux1 = [];
    aux8 = [];
    results = [results; s];
    results(i).Info = ['C3E With Squared Loss Function'];
    EvalEns = [];
    EvalC3E = [];
    
    % when 'folds' is a negative value, its positive value will be the fold
    % and totalFolds will be 1.
    if (folds < 0)
        totalFolds = 1;
        fold = folds*(-1);
    end 
    
    for j = 1:totalFolds
        
        % run every fold (j) up to 1Tfold
        if (folds < 0)
            fprintf('\n');
            fprintf('------------------------------------------------------------------\n');
            fprintf('------------------------------ FOLD %i ----------------------------\n', fold);
            fprintf('------------------------------------------------------------------\n');
            [labels, piSet, SSet, nFiles] = runJavaMLA_SSL(data_path, data.(nData{numDatasets(i)}), fold, numObjFold(i), validation, mc, nInst, strmc, strni, iter);
        else
            fprintf('\n');
            fprintf('------------------------------------------------------------------\n');
            fprintf('------------------------------ FOLD %i ----------------------------\n', j);
            fprintf('------------------------------------------------------------------\n');
            [labels, piSet, SSet, nFiles] = runJavaMLA_SSL(data_path, data.(nData{numDatasets(i)}), j, numObjFold(i), validation, mc, nInst, strmc, strni, iter);
        end 
                              
        save([pwd,'/results/labels.mat'], 'labels');
        save([pwd,'/results/piSet.mat'], 'piSet');
        if (validation > 0)
            save([pwd,'/results/SSetVal.mat'], 'SSet');
            if (strcmp(iter,'0'))
                save([pwd,'/results/SSetVal.mat'], 'SSet');
            end
        else
            save([pwd,'/results/SSet.mat'], 'SSet');
            if (strcmp(iter,'0'))
                save([pwd,'/results/SSetInit.mat'], 'SSet');
            end
        end
        
        %%%% AQUI DETERMINA A CLASSE A PARTIR DAS DIST. PROB. CLASSES
        [~, piSetLabel] = max(piSet');
        
        % compute accuracy, balanced accuracy, precision, sensitivity, specificity, f_measure, gmean
        % storing in a matrix (first and second columns are the fold and class, respectively)
        % (zeros in the second column are general results - considering all classes)
        [sumResults, resultsPerClass, ~] = evaluateClassifier(labels, piSetLabel');
        if (totalFolds == 1)
            EvalEns = [EvalEns; [[fold*ones(size(resultsPerClass,1),1),resultsPerClass(:,1),zeros(size(resultsPerClass,1),1),resultsPerClass(:,2:8)];[fold,0,sumResults']]];
        else
            EvalEns = [EvalEns; [[j*ones(size(resultsPerClass,1),1),resultsPerClass(:,1),zeros(size(resultsPerClass,1),1),resultsPerClass(:,2:8)];[j,0,sumResults']]];
        end

        %accuracy = 100*mean(labels==piSetLabel');
        %accuracy = sumResults(1)*100; % accuracy
        accuracy = sumResults(2)*100; % balanced accuracy
        %accuracy = sumResults(3)*100; % precision
        %accuracy = sumResults(4)*100; % sensitivity
        %accuracy = sumResults(6)*100; % f_measure
        %accuracy = sumResults(8)*100; % relError
        
        aux3 = zeros(size(alpha,1)+1,size(numIter,1)+1);
        aux3(1,2:size(numIter,1)+1) = numIter';
        aux3(2:size(alpha,1)+1,1) = alpha;
        aux4 = aux3;
        aux2 = [];
        
        t0 = tic;
        
        y = [];
        
        for m = 1:size(alpha,1)

           fprintf('%s - %s - F: %i - alpha: %1.4f - nit: %i\n', datestr(now), data.(nData{numDatasets(i)}), j, alpha(m), max(numIter));
             
           if (C3EVersion == 0) 
               [accuracyRef, ~, vObj] = C3E(piSet, SSet, labels, alpha(m), 1, numIter);
           else 
               if (C3EVersion == 1)
                   [accuracyRef, ~, vObj, ylabel, y] = C3ESLWrappered(piSet, SSet, labels, alpha(m), numIter);
               else 
                   accuracyRef = zeros(max(numIter),1);
                   vObj = zeros(max(numIter),1);
                   for k = 1:max(numIter)
                       [vAcc, vObjF, ~, ~] = C3ESL(piSet, SSet, labels, alpha(m), k);
                       accuracyRef(k) = vAcc;
                       vObj(k) = vObjF;
                   end
               end 
           end 

           if (size(alpha,1)==1) && (size(numIter,1)==1)
               % compute accuracy, balanced accuracy, precision, sensitivity, specificity, f_measure, gmean
               % storing in a matrix (first and second column are the fold and class, respectively)
               % (zeros in the second column are general results - considering all classes)
               [sumResults, resultsPerClass, ~] = evaluateClassifier(labels, ylabel');
               if (totalFolds == 1)
                   EvalC3E = [EvalC3E; [[fold*ones(size(resultsPerClass,1),1),resultsPerClass(:,1),zeros(size(resultsPerClass,1),1),resultsPerClass(:,2:8)];[fold,0,sumResults']]];
               else
                   EvalC3E = [EvalC3E; [[j*ones(size(resultsPerClass,1),1),resultsPerClass(:,1),zeros(size(resultsPerClass,1),1),resultsPerClass(:,2:8)];[j,0,sumResults']]];
               end
               %save([pwd,'/results/CMC3E', sFCAE,'.mat'],'CMC3E');
           end

           aux3(m+1,2:size(numIter,1)+1) = accuracyRef';
           aux4(m+1,2:size(numIter,1)+1) = vObj';
           aux2 = [aux2, accuracyRef'];
        end
 
        aux8 = [aux8; toc(t0)];
        
        nResults = fieldnames(results(1));
        results(i).NameData = data.(nData{numDatasets(i)});
        results(i).Before = [results(i).Before; accuracy];
        results(i).(nResults{j+1}) = aux3;
        results(i).Obj = aux4;
        aux1 = [aux1; aux2];
        
        inf = ['Ensemble Java - Fold ', int2str(j), ': ', ' '];
        results(i).Info = [results(i).Info; [inf, repmat('.', 1, size(results(i).Info(1,:),2) - size(inf,2))]];
    end
    
    aux6 = [mean(aux1); var(aux1); min(aux1); max(aux1)];

    results(i).Time = aux8;

    % COMENTADO PORQUE DÁ PAU QUANDO USA GRID-SEARCH NO VALIDATION SET
    %results(i).M = results(i).F1;
    %results(i).M(2:size(results(i).M,1),2:size(results(i).M,2)) = reshape(aux6(1,:), size(numIter,1), size(alpha,1))';

    %results(i).V = results(i).F1;
    %results(i).V(2:size(results(i).V,1),2:size(results(i).V,2)) = reshape(aux6(2,:), size(numIter,1), size(alpha,1))';

    %results(i).Mi = results(i).F1;
    %results(i).Mi(2:size(results(i).Mi,1),2:size(results(i).Mi,2)) = reshape(aux6(3,:), size(numIter,1), size(alpha,1))';

    %results(i).Ma = results(i).F1;
    %results(i).Ma(2:size(results(i).Ma,1),2:size(results(i).Ma,2)) = reshape(aux6(4,:), size(numIter,1), size(alpha,1))';
    
    results(i).EvalEns = EvalEns;
    results(i).EvalC3E = EvalC3E;
    
    if (saveFile > 0)
        % está com algum bug
        % save([pwd,'/results/', nFiles.piSet(7:size(nFiles.piSet,2)-4), '-', nFiles.SSet(6:size(nFiles.SSet,2)-4), '.mat'], results);
    end
end
