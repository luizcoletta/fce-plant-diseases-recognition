function [vAcc, vObj, ylabel, y] = C3ESL(piSet, SSet, trueLabel, alpha, I)
    % *************************************************************************
    % C3ESL: C3E based on a Squared Loss function. It assumes an optimization 
    % procedure that takes as input class membership estimates from existing 
    % classifiers (piSet), as well as a similarity matrix (SSet) from a cluster 
    % ensemble induced on the target data, to provide a consolidated 
    % classification of the objects in the target data (y).
    %
    % piSet:     a matrix nxm, where rows are objects and colunms are class  
    %            probability distributions;
    % SSet:      a nxn similarity matrix;
    % trueLabel: a size-n vector with the true labels;
    % alpha:     a real-valued number; 
    % I:         an integer number.
    %
    % vAcc:   a real-valued number that is the accuracy;
    % vObj:   a real-valued number that is the output of the objective
    %         function
    % ylabel: a size-n vector with the estimated labels;
    % y:      a matrix nxm, where lines are objects and colunms are refined 
    %         class probability distributions;
    %
    % More Details: 
    % http://link.springer.com/chapter/10.1007/978-3-642-21557-5_29
    % http://dl.acm.org/citation.cfm?id=2601435
    % http://www.inderscience.com/offer.php?id=69288
    % http://content.iospress.com/articles/integrated-computer-aided-engineering/ica485
    % http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6984832
    % http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6855895
    %
    % Author: Luiz F. S. Coletta (luizfsc@tupa.unesp.br) - 07/03/12
    % Update: Luiz F. S. Coletta - 28/09/16
    % *************************************************************************

    c = size(piSet,2);    % number of classes
    N = size(piSet,1);    % number of objects

    % initialization of class assignment probability vector
    y=ones(N,c);
    y=y./repmat(sum(y,2),1,c);

    % C3E-SL algorithm
    for j = 1:I

        diffi = 1:N; 
        for i = 1:N
            
            t1 = SSet(i,diffi(diffi~=i));
   
            p1 = sum((t1'*ones(1,c)).*y(diffi(diffi~=i),:));
            p2 = sum(t1);
            
            y(i,:) = (piSet(i,:) + (2*alpha*p1)) / (1 + 2*alpha*p2);
        end 
    end 

    % computes objective function
    vObj = evalObj(piSet, SSet, alpha, y);
    
    % computes accuracy
    [~, ylabel] = max(y');
    ylabel = ylabel';
    vAcc = 100*mean(trueLabel==ylabel);

end

function [value] = evalObj(piSet, SSet, alpha, y)
   
    p1 = (sum(sum((y-piSet).^2)))/2;

    % returns all combinations to map SSet entries
    allComb = combnk(1:size(piSet,1),2);
    
    % gets the values of SSet entries
    allEnt = SSet(sub2ind(size(SSet),allComb(:,1),allComb(:,2)));
    
    t1 = sum((y(allComb(:,1),:)-y(allComb(:,2),:)).^2');
    t2 = sum((allEnt.*t1'));
    
    p2 = alpha*(t2/2);
    
    value = p1 + p2;
end 

