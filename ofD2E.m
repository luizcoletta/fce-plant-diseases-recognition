function val = ofD2E(opar, params, NF)

nameFile = [NF,'-fobj.mat'];

x = params(1); % al
y = params(2); % ni

% opar(6) % val
% opar(5) % nData
% opar(8) % sizeValSet
% opar(4) % fold 
% opar(7) % missingClass
% opar(9:size(opar,2)) % newInst

res = feval('runSemiSuperLearning', 0, opar(5), opar(6), opar(8), opar(4), x, y, opar(7), opar(9:size(opar,2)), '0'); 
%res = feval('runTestsSS', x, y, opar(6), opar(5), opar(8), opar(4), opar(7), opar(9:size(opar,2)), '0', 0, 0);   

val = 100 - res.M(2:size(res.M,1),2);

path_results = '/results/';

path_resSum = [pwd, path_results, nameFile];
if (exist(path_resSum, 'file'))
    resFObj = load(path_resSum, '-mat');
    resSum = resFObj.resSum;
else
    r = struct('F1',[],'F2',[],'F3',[],'F4',[],'F5',[],'F6',[],'F7',[],'F8',[],'F9',[],'F10',[]);
    resSum = [r;r;r;r;r;r;r;r;r;r];
end

if (opar(4) < 0)
   opar(4) = opar(4)*(-1);    
end 

dataNames = fieldnames(resSum(opar(5)));
r = resSum(opar(5)).(dataNames{opar(4)});
r = [r; [x, y, val, opar(4), opar(5)]];
resSum(opar(5)).(dataNames{opar(4)}) = r;
save([pwd, path_results, nameFile], 'resSum');
