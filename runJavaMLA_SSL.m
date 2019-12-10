function [labels, piSet, SSet, resFiles] = runJavaMLA_SSL(dataPath, nameData, fold, totObjValSet, validation, mc, nInst, strmc, strni, iter)
                                                                
resFiles = [];

% NB - J48 - IB5 - SVM
typeClaEns = [0,0,0,1];

% 1: cluster ensemble strategy 1 (old)
%    estimating k, building until three partitions with k*theta, k*theta*2
%    and k*theta*3 using the same space of features.
% 2: cluster ensemble strategy 2 (new)
%    selecting randomly subset of features, building partitions from each
%    subset; estimating k in each one, combining the best partitions.
% 3: cluster ensemble strategy 3 (for tweet sentiment analysis)
%    by using k-medoids + cosine similarity (ELKI tool).
strategyCluEns = 2;

% number of partitions (1, 2, 3, ...)
% if 0 it won't run the cluster ensemble neither C3E!
typeCluEns = 4;

% multiplier for strategy 1 OR
% to get the size of the subset of features in strategy 2
% 1=10%; 2=20%, 3=30%, ...
theta = 2;

if (validation == 0)
   sFCAE = ['_', nameData(1:size(nameData,2)-5), num2str(fold), strmc, strni, iter, num2str(typeClaEns(1)), num2str(typeClaEns(2)), num2str(typeClaEns(3)), num2str(typeClaEns(4))];
   sFCUE = ['_', nameData(1:size(nameData,2)-5), num2str(fold), strmc, strni, iter, num2str(strategyCluEns), num2str(theta), num2str(typeCluEns)];
else
   sFCAE = ['_', nameData(1:size(nameData,2)-5), 'R', num2str(fold), strmc, strni, iter, num2str(validation), num2str(typeClaEns(1)), num2str(typeClaEns(2)), num2str(typeClaEns(3)), num2str(typeClaEns(4))];
   sFCUE = ['_', nameData(1:size(nameData,2)-5), 'R', num2str(fold), strmc, strni, iter, num2str(validation), num2str(strategyCluEns), num2str(theta), num2str(typeCluEns)];
end

labelFile = [pwd, '/results/labels', sFCAE, '.dat'];
piSetFile = [pwd, '/results/piSet', sFCAE, '.dat'];
if (exist(labelFile, 'file') && exist(piSetFile, 'file'))
    typeClaEns(1) = 0;
    typeClaEns(2) = 0;
    typeClaEns(3) = 0;
    typeClaEns(4) = 0;
end 

% if iterations from 'actLearning', update coassociation matrix (without 
% build another cluster ensemble
SSetFile = [pwd, '/results/SSet', sFCUE, '.dat'];
if (exist(SSetFile, 'file'))
    typeCluEns = 0;
else
    if (~strcmp(iter,'0'))
        if (validation > 0)
            SWrk = load([pwd,'/results/SSetInitVal.mat']);
        else
            SWrk = load([pwd,'/results/SSetInit.mat']);
        end
        SSetNew = SWrk.('SSet');
        clear SWrk
        SSetNew(:,nInst) = [];
        SSetNew(nInst,:) = [];
        dlmwrite([pwd,'/results/SSet', sFCUE,'.dat'], SSetNew, 'delimiter', '\t', 'precision', '%1.2f');
        typeCluEns = 0;
    end
end

claEns = java.util.ArrayList;
claEns.add(java.lang.Integer(typeClaEns(1))); % NB
claEns.add(java.lang.Integer(typeClaEns(2))); % J48
claEns.add(java.lang.Integer(typeClaEns(3))); % IB5
claEns.add(java.lang.Integer(typeClaEns(4))); % SVM

%  0: the train set will contain all classes
% >0: the train set will not contain the class from this index
missingClasses = java.util.ArrayList;
missingClasses.add(java.lang.Integer(mc));

%  <0: there are no new classes
% >=0: there are new classes
newInst = java.util.ArrayList;
for i = 1:size(nInst,2)
    newInst.add(java.lang.Integer(nInst(i)));
end

% load java classes
javaaddpath({[pwd, '/javamla/lib/weka-3.9.1-SNAPSHOT.jar']});
javaaddpath({[pwd, '/javamla/lib/elki-bundle-0.7.1.jar']});
javaaddpath({[pwd, '/javamla/']});

ensemble = RunEnsembleHoldout;

% Run JAVAMLA to generate classifier and clusterer ensembles. Results will
% be saved in 'piSet.dat' and 'SSet.dat', respectively. The file 'labels.dat'
% contains the ground truth. For validation the files' name will be with 'R';
%
% PARAMETERS:
%   dataPath        -> path of dataset
%   path_results    -> path in which results will be saved
%   fold            -> cross validation fold (if fold = 0 is used only one fold (train and test are the same set))
%   totObjValSet    -> number of objects in the validation set
%   claEns          -> classifier ensemble
%   strategyCluEns  -> cluster ensemble strategy 1 or 2 or ...
%   theta           -> multiplier (for the number of clusters k)
%   typeCluEns      -> if 1, 1 kmeans(k*theta); if 2, 2 kmeans(k*theta*2); if 3, 3 kmeans(k*theta*3)
%   missingClasses  -> missing classes in the train set
%   validation      -> 0: for testing (to build the test and train sets); 1/2 for validation
%   newInst         -> <0: there are no new objects; >=0: there are new objects (from indexes numbers)
%   printInfo       -> if 0, it does not print results; otherwise it does
%   iter            -> number of iteration of the algorithm for identifying new classes
res = ensemble.RunEnsembleSSL([pwd, dataPath, nameData], [pwd, '/results/'], fold, totObjValSet, claEns, strategyCluEns, theta, typeCluEns, missingClasses, validation, newInst, 1, iter, sFCAE, sFCUE);

try
    % labels-piSet Name: DatasetName + "R" for validation + Fold + MissingClass + NewInstances + Iteration + If "R", ValidationSet + EnsembleClassVector
    % SSet Name: DatasetName + "R" for validation + Fold + MissingClass + NewInstances + Iteration + If "R", ValidationSet + ClusterEnsembleStrategy + Multiplier(strat1)/SizeSubsetFeatures(strat2) + NumberOfClusters 
    resFiles.labels = res.get(0);
    resFiles.piSet = res.get(1);
    resFiles.SSet = res.get(2);
    
    labels = [];
    piSet = [];
    SSet = [];
    
    labels = load([pwd, '/results/', resFiles.labels]);
    piSet = load([pwd, '/results/', resFiles.piSet]);
    SSet = load([pwd, '/results/', resFiles.SSet]);
    
    [~, piSetLabel] = max(piSet');
    dlmwrite([pwd, '/results/piSetLabel', sFCAE, '.dat'], piSetLabel', 'delimiter', '\t');
    %save([pwd,'/results/labels.mat'],'labels');
    %save([pwd,'/results/piSet.mat'],'piSet'); 
catch err
end
