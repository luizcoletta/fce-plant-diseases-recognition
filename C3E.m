function [vAccuracy, vCount, vObj] = C3E(piSet, SSet, true_label, alpha, lambda, numiter)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for OAC3
% @Ayan Acharya, Date: 08.01.2011
% contact ID: masterayan@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% piSet:      n X k, n: number of instances, k: number of classes; class probability assignment of test instances from classifier ensemble
% SSet:       n X n; similarity matrix obtained from cluster ensemble
% true_label: n X 1; true labels of the test data
% alpha:      co-efficient of the second term
% lambda:     value of Lagrangian multiplier
% numiter:    maximum number of iterations allowed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs:
% accuracy:   accuracy on test data
% y:          refined final class probability assignment
% ylabel:     class labels of test data as obtained by C3E
% count:      number of iterations performed before percentage difference in objective function converges to 'precsn'
% tm2:        computation time for optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

format long;

ncl = size(piSet,2);    % number of classes
N   = size(piSet,1);    % number of data points

%declaring constants
errctrlr = 0.000000001; % to avoid log(zero)

%uniform class assignment for unlabeled points
ind=find(diag(piSet*piSet')==0);
if(isempty(ind)==0)
    piSet(ind,:)=1/ncl;
end
piSet=piSet+errctrlr;
piSet=piSet./repmat(sum(piSet,2),1,ncl);

%initialization of class assignment probability vector
yl=ones(N,ncl); %left copy
yl=yl./repmat(sum(yl,2),1,ncl);
yr=yl;          %right copy

%obj = evaluate_obj(piSet, SSet, yl, yr, lambda, alpha);
obj = 0;

%[~,ind] = max(piSet,[],2);
[xxx,ind] = max(piSet,[],2);
Clsacc  = 100*mean(true_label==ind);

tm1 = cputime;
count=1;

gammar  = repmat((alpha*sum(SSet,1))',1,ncl);
gammal  = repmat(alpha*sum(SSet,2),1,ncl);
delta   = SSet./repmat(sum(SSet,1), N, 1);

% disp('From C3E');

vAccuracy = [];
vCount = [];
vObj = [];

countNumIter = 1;

MAXCOUNT = max(numiter);

while(count<=MAXCOUNT)
    yr     = (piSet+gammar.*(delta*yl)+lambda*yl)./(1+gammar+lambda);
    
    if (numiter(countNumIter) ~= count)
        %obj = evaluate_obj(piSet, SSet, yl, yr, lambda, alpha);
        obj = 0;
    end
    if (numiter(countNumIter) == count)
        
        yl=yl./repmat(sum(yl,2),1,ncl);
        yr=yr./repmat(sum(yr,2),1,ncl);
        y=[yl+yr]/2;
        
        %obj = evaluate_obj(piSet, SSet, y, y, lambda, alpha);
        obj = 0;
        
        [ymax ylabel]=max(y');
        accuracy = 100*mean(true_label==ylabel');
        ylabel=ylabel';
        tm2 = cputime-tm1;
        
        prox = (sum(sum(corrcoef(piSet,y)))-2)/2;
        
        vAccuracy = [vAccuracy; accuracy];
        vCount = [vCount, count];
        vObj = [vObj; obj];
        
        countNumIter = countNumIter + 1;
    end
    
    gradyr = 1+log(yr+errctrlr);
    temp   = (gammal.*(delta*gradyr)+lambda*gradyr)./(gammal+lambda);
    yl     = exp(temp-1);
    
    %fprintf('\nCount %i %d %d', count, yr, yl);
    
    count=count+1;
end
end

function [objval]=evaluate_obj(piSet, SSet, yl, yr, lambda, alpha)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computes the objective function
% @Ayan Acharya, Date: 08.01.2011
% contact ID: masterayan@gmail.com

% Input:
% piSet: n X k, n: number of instances, k: number of classes; probablity vector corresponding to each instance obtained from classifier ensemble
% SSet: n X n; similarity matrix from the cluster ensemble
% yl: n X k; left copies
% yr: n X k; right copies
% lambda: parameter $\lambda$ in OAC3
% alpha: parameter $\alpha$ in OAC3

% Output:
% objval: value of the objective function in OAC3 with Lagrangian multiplier introduced for left and right copies of class assignment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=size(SSet,1);

term11=sum(sum(piSet.*log(piSet)));
term12=sum(sum(yr.*log(yr)));
gradyr=1+log(yr);
term13=sum(sum(piSet.*gradyr));
term14=sum(sum(yr.*gradyr));
term1=term11-term12-term13+term14;

Srow=sum(SSet,2)-[diag(SSet)];
Scol=[sum(SSet,1)-diag(SSet)']';
phiyl=sum(yl.*log(yl),2);
phiyr=sum(yr.*log(yr),2);
term21=Srow'*phiyl;
term22=Scol'*phiyr;
prodylgradyr=yl*gradyr';
diagval=diag(prodylgradyr);
prodylgradyr=prodylgradyr-diag(diagval);
term23=sum(sum(SSet.*prodylgradyr));
prodyrgradyr=yr*gradyr';
diagval=diag(prodylgradyr);
term24=Scol'*diagval;
term2=term21-term22-term23+term24;

term31=lambda*sum(sum(piSet.*log(piSet)));
term32=lambda*sum(sum(yr.*log(yr)));
term33=lambda*sum(sum(yl.*gradyr));
term34=lambda*sum(sum(yr.*gradyr));
term3=term31-term32-term33+term34;

objval=term1+alpha*term2+term3;

end
