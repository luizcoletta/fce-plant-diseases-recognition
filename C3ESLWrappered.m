function [vAccuracy, vCount, vObj, ylabel, y] = C3ESLWrappered(piSet, SSet, true_label, alpha, niter)
% *************************************************************************
% C3E_SL: C3E algorithm using Squared Loss function
%
% Author: Luiz F. S. Coletta (luiz.fersc@gmail.com) - 07/03/12
% Update: Luiz F. S. Coletta - 01/04/14
% *************************************************************************

ncl = size(piSet,2);    % number of classes
N = size(piSet,1);      % number of data points
errctrlr = 0.000000001; % to avoid log(zero)

% uniform class assignment for unlabeled points
ind=find(diag(piSet*piSet')==0);
if(isempty(ind)==0)
    piSet(ind,:)=1/ncl;
end
piSet=piSet+errctrlr;
piSet=piSet./repmat(sum(piSet,2),1,ncl);

% initialization of class assignment probability vector
y=ones(N,ncl);
y=y./repmat(sum(y,2),1,ncl);

vAccuracy = [];
vCount = [];
vObj = [];
count = 0;
countNumIter = 1;

MAXCOUNT = max(niter);

par = 0;

while(count<MAXCOUNT)
    
    % compute new y
    if (par == 0)
        aux = 1:size(y,1); 
        for i = 1:size(SSet,1)

            % iterative version
            %p1 = 0;
            %p2 = 0;
            %for j = 1:size(SSet,2)
            %    if (i ~= j)
            %        p1 = p1 + SSet(i,j)*y(j,:);
            %        p2 = p2 + SSet(i,j);
            %    end 
            %end
            %y_new(i,:) = (piSet(i,:) + 2*alpha*p1) / (1 + 2*alpha*p2);

            p1 = sum((SSet(i,aux(aux~=i))'*ones(1,size(piSet,2))).*y(aux(aux~=i),:));
            p2 = sum(SSet(i,aux(aux~=i)));
            y(i,:) = (piSet(i,:) + 2*alpha*p1) / (1 + 2*alpha*p2);
        end 
    else 
%         aux = 1:size(y,1); 
%         y_copy = y;
%         
%         initParComp('start');
%         
%         %tic
%   
%         parfor i = 1:size(SSet,1)
%             p1 = sum((SSet(i,aux(aux~=i))'*ones(1,size(piSet,2))).*y_copy(aux(aux~=i),:));
%             p2 = sum(SSet(i,aux(aux~=i)));
%             y(i,:) = (piSet(i,:) + 2*alpha*p1) / (1 + 2*alpha*p2);
%         end 
%         
%         %fprintf('%1.4f\n',toc);
%         
%         initParComp('stop');
    end 
   
    % compute objective function
    %obj = evaluate_obj(piSet, SSet, alpha, y);
    obj = 0;

    count=count+1;

    if (niter(countNumIter) == count) 
       [ymax ylabel]=max(y');
       
       %vAccuracy = [vAccuracy; 100*mean(true_label==ylabel')];
       
       % compute accuracy, balanced accuracy, precision, sensitivity, specificity, f_measure, gmean
       % storing in a matrix (first and second column are the fold and class, respectively)
       % (zeros in the second column are general results - considering all classes)
       [sumResults] = evaluateClassifier(true_label,ylabel');
         
       %vAccuracy = [vAccuracy; sumResults(1)*100]; % accuracy
       vAccuracy = [vAccuracy; sumResults(2)*100]; % balanced accuracy
       %vAccuracy = [vAccuracy; sumResults(3)*100]; % precision
       %vAccuracy = [vAccuracy; sumResults(4)*100]; % sensitivity
       %vAccuracy = [vAccuracy; sumResults(6)*100]; % f_measure
       %vAccuracy = [vAccuracy; sumResults(8)*100]; % relError
       
       vCount = [vCount; count];
       vObj = [vObj; obj];
       countNumIter = countNumIter + 1;
    end 
end
end 

function [value]=evaluate_obj(piSet, SSet, alpha, y)
   
    p1 = (sum(sum((y-piSet).^2)))/2;

    c = combnk(1:size(piSet,1),2);
    s = [1:1:size(piSet,1);1:1:size(piSet,1)]';
    
    t11 = SSet(sub2ind(size(SSet),c(:,1),c(:,2)));
    t12 = sum((y(c(:,1),:)-y(c(:,2),:)).^2');
    t1 = sum((t11.*t12'))*2;
    t2 = sum(sum((y(s(:,1),:)-y(s(:,2),:)).^2'));
    p2 = alpha*((t1+t2)/2);
    
    value = p1 + p2;
    
    %fprintf('C3E-SL Obj. Func.: %1.4f\n', value);
end 

% function [nWorkers]=initParComp(msg)
%    
%     nWorkers = 0;
%     if (strcmp(msg,'start')) 
%         if (matlabpool('size') == 0)
%             myCluster = parcluster('local');
%             nWorkers = myCluster.NumWorkers;
%             matlabpool('local', nWorkers);
%         end    
%     end 
%     if (strcmp(msg,'stop'))
%         matlabpool close;
%     end
%   
% end 
