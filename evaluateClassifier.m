function [sumResults, resultsPerClass, confusionMatrix] = evaluateClassifier(true_labels,labels)
% *************************************************************************
% evaluateClassifier:  provides the following measures: accuracy, balanced  
%                      accuracy, precision, sensitivity, specificity, f_measure, 
%                      and gmean.  
%
% Example:  [sumResults, resultsPerClass] = evaluateClassifier(true_labels,labels)
%
% Author: Luiz F. S. Coletta (luiz.fersc@gmail.com) - 21/03/14
% Update: Luiz F. S. Coletta - 02/04/14
% *************************************************************************

availLabels = unique(true_labels); % get labels names
nClasses = size(availLabels,1);    % get number of classes
errVal = 0.000000001;

% compute the confusion matrix
% -----------------------------------
%         | Predicted by Classifier
% -----------------------------------           
% Correct |  
% Class   | 
% Labels  |
confusionMatrix = zeros(nClasses,nClasses);
for i = 1:nClasses
    idx = (true_labels==availLabels(i));
    for j = 1:nClasses
        confusionMatrix(i,j) = sum(labels(idx)==availLabels(j));
    end
end

% count tp, tn, fp, fn per class (store in a matrix in which each line is a class)
sumPerClass = zeros(nClasses,4);
for i = 1:nClasses
    
   curClass = availLabels(i); 
    
   idxP = availLabels==curClass;
   idxN = availLabels~=curClass;
   
   iClassAux1 = confusionMatrix(i,:);
   iClassAux2 = confusionMatrix(:,i);
 
   tp = iClassAux1(idxP);
   tn = sum(sum(confusionMatrix(idxN,idxN)));
   fp = sum(iClassAux2(idxN));
   fn = sum(iClassAux1(idxN));
   
   sumPerClass(i,:) = [tp,tn,fp,fn]; 
end

% compute performance measures by each class 
precisionPerClass = zeros(nClasses,1);
sensitivityPerClass = zeros(nClasses,1);
specificityPerClass = zeros(nClasses,1);
f_measurePerClass = zeros(nClasses,1);
gmeanPerClass = zeros(nClasses,1);
balAccPerClass = zeros(nClasses,1);
relErrorPerClass = zeros(nClasses,1);

for i = 1:nClasses
   precisionPerClass(i)   = sumPerClass(i,1)/(sumPerClass(i,1)+sumPerClass(i,3)+errVal);  % precision = tp/(tp+fp);
   sensitivityPerClass(i) = sumPerClass(i,1)/(sumPerClass(i,1)+sumPerClass(i,4));         % recall = tp_rate = tp/(tp+fn);
   specificityPerClass(i) = sumPerClass(i,2)/(sumPerClass(i,2)+sumPerClass(i,3));         % tn_rate = tn/(tn+fp);
   f_measurePerClass(i)   = 2*((precisionPerClass(i)*sensitivityPerClass(i))/(precisionPerClass(i)+sensitivityPerClass(i)+errVal)); % f_measure = 2*((precision*recall)/(precision + recall)); 
   gmeanPerClass(i)       = sqrt(sensitivityPerClass(i)*specificityPerClass(i));          % gmean = sqrt(tp_rate*tn_rate);
   balAccPerClass(i)      = (sensitivityPerClass(i)/2)+(specificityPerClass(i)/2);        % 0.5*((tp/(tp+fn))+(tn/(tn+fp)))
   
   e1 = sumPerClass(i,3)/(sum(sumPerClass(i,:))-(sumPerClass(i,1)+sumPerClass(i,4)));     % fp/(n-ni)
   e2 = sumPerClass(i,4)/((sumPerClass(i,1)+sumPerClass(i,4)));                           % fn/ni
  
   relErrorPerClass(i)    = e1 + e2;
end

%class_param = calc_class_param(labels,true_labels)

% compute general measures (considering all classes)
accuracy    = sum(diag(confusionMatrix))/sum(sum(confusionMatrix));
balAcc      = sum(balAccPerClass)/nClasses; % balAcc = 1-(sum(relErrorPerClass)/(2*nClasses));
precision   = sum(precisionPerClass)/nClasses;
sensitivity = sum(sensitivityPerClass)/nClasses;
specificity = sum(specificityPerClass)/nClasses;
f_measure   = sum(f_measurePerClass)/nClasses;
gmean       = sum(gmeanPerClass)/nClasses;
relError    = sum(relErrorPerClass)/nClasses;

resultsPerClass = [availLabels, balAccPerClass, precisionPerClass, sensitivityPerClass, specificityPerClass, f_measurePerClass, gmeanPerClass, relErrorPerClass];
sumResults = [accuracy; balAcc; precision; sensitivity; specificity; f_measure; gmean; relError];
