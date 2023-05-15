function MOEAD(Global)
% <algorithm> <M>
% Multiobjective evolutionary algorithm based on decomposition
% type --- 1 --- The type of aggregation function
%类型--1--聚合函数的类型
%------------------------------- Reference --------------------------------
% Q. Zhang and H. Li, MOEA/D: A multiobjective evolutionary algorithm based
% on decomposition, IEEE Transactions on Evolutionary Computation, 2007,
% 11(6): 712-731.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    type = Global.ParameterSet(1);

    %% Generate the weight vectors 生成权重向量
    [W,Global.N] = UniformPoint(Global.N,Global.M);%生成均匀分布的权重向量N个，设为W，M个目标
    T = ceil(Global.N/10);%ceil 向上取整，邻居T

    %% Detect the neighbours of each solution 设置邻居向量
    B = pdist2(W,W);%计算距离
    [~,B] = sort(B,2);%1表示列排序，2表示行排序，对B升序排
    B = B  (:,1:T);%取B的1-T列，即B最小的前T个为邻居
    
    %% Generate random population
    Population = Global.Initialization();
    Z = min(Population.objs,[],1);%返回种群每列最小元素，生成每列最小行向量，作为最小点
    Z1 = max(Population.objs,[],1);%case 10 11
    %%
    angle2  = acos(dot(W(:,:),W(:,:))./(norm(W(:,:))*(norm(W(:,:)))));
    angle   = sum(sum(mink(angle2,Global.M)))/Global.M;
    
    %% Optimization
    while Global.NotTermination(Population)
        % For each solution
        for i = 1 : Global.N     %N  population size
            % Choose the parents
            P = B(i,randperm(size(B,2)));%size看数组长度，randperm随机排列

            % Generate an offspring
            Offspring = GAhalf(Population(P(1:2)));

            % Update the ideal point
            Z = min(Z,Offspring.obj);
            Z1 = max(Population.objs,[],1);
            size(Z1)%case 10 11
            % Update the neighbours
            switch type
                case 1
                    % PBI approach
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                    g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                    g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                case 2
                    % Tchebycheff approach
                    g_old = max(abs(Population(P).objs-repmat(Z,T,1)).*W(P,:),[],2);
                    g_new = max(repmat(abs(Offspring.obj-Z),T,1).*W(P,:),[],2);
                case 3
                    % Tchebycheff approach with normalization
                    Zmax  = max(Population.objs,[],1);
                    g_old = max(abs(Population(P).objs-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(P,:),[],2);
                    g_new = max(repmat(abs(Offspring.obj-Z)./(Zmax-Z),T,1).*W(P,:),[],2);
                case 4
                    % Modified Tchebycheff approach
                    g_old = max(abs(Population(P).objs-repmat(Z,T,1))./W(P,:),[],2);
                    g_new = max(repmat(abs(Offspring.obj-Z),T,1)./W(P,:),[],2);
                case 5
                    %Double PBI ok
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                    %angle1 angle2
                    %g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                    %g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                    %g_old   = normP.*CosineP + 4.6*(normP.*sqrt(1-CosineP.^2)).^2;
                    g_old   = normP.*CosineP + 4.6*(normP.*sqrt(1-CosineP.^2)).^2+0.1*(normP.*sqrt(1-CosineP.^2)).^4;
                    g_new   = normO.*CosineO + 4.6*(normO.*sqrt(1-CosineO.^2)).^2+0.1*(normO.*sqrt(1-CosineO.^2)).^4;
                case 6
                    % APS ok
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                    angle   = 1+(10-1)*Global.gen/Global.maxgen;
                    g_old   = normP.*CosineP + angle*normP.*sqrt(1-CosineP.^2);
                    g_new   = normO.*CosineO + angle*normO.*sqrt(1-CosineO.^2);
                case 7
                    % SPS wrong
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                    bi      = max(W(P,:),[],2)-min(W(P,:),[],2);
                    angle   = exp(4*bi);
                    g_old   = normP.*CosineP + angle.*normP.*sqrt(1-CosineP.^2);
                    g_new   = normO.*CosineO + angle.*normO.*sqrt(1-CosineO.^2);
                case 8
                    % PSF ok
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                    g_old = min((max(abs(Population(P).objs-repmat(Z,T,1))./W(P,:),[],2)+10*normP.*sqrt(1-CosineP.^2)),[],2);
                    %g_new = min((max(abs(Population(P).objs-repmat(Z,T,1))./W(P,:),[],2)+10*normP.*sqrt(1-CosineP.^2)),[],2);
                    %g_new = max(repmat(abs(Offspring.obj-Z),T,1)./W(P,:),[],2)+10*normO.*sqrt(1-CosineO.^2);
                    g_new = min((max(repmat(abs(Offspring.obj-Z),T,1)./W(P,:),[],2)+10*normO.*sqrt(1-CosineO.^2)),[],2);
                case 9
                    % MSF ok 自适应angle没打
                    angle = (1-Global.gen/Global.maxgen).*min(W(P,:),[],2);
                    a     = max(abs(Population(P).objs-repmat(Z,T,1))./W(P,:),[],2).^(angle+1);
                    b     = min(abs(Population(P).objs-repmat(Z,T,1))./W(P,:),[],2).^angle;
                    g_old = (a./b);  
                    a_new = max(repmat(abs(Offspring.obj-Z),T,1)./W(P,:),[],2).^(angle+1);
                    b_new = min(repmat(abs(Offspring.obj-Z),T,1)./W(P,:),[],2).^angle;
                    g_new = (a_new./b_new);
                case 10
                    % iPBI ok 反向
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((repmat(Z1,T,1)-Population(P).objs).^2,2));
                    normO   = sqrt(sum((Z1-Offspring.obj).^2,2));
                    CosineP = sum((repmat(Z1,T,1)-Population(P).objs).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(Z1-Offspring.obj,T,1).*W(P,:),2)./normW./normO;
                    g_old1   = normP.*CosineP - 5*normP.*sqrt(1-CosineP.^2);
                    g_new1   = normO.*CosineO - 5*normO.*sqrt(1-CosineO.^2);
                case 11
                    %rTCH 效果很差 但是感觉没打错
                    g_old1 = min(abs(repmat(Z1,T,1)-Population(P).objs)./W(P,:),[],2);
                    g_new1 = min(repmat(abs(Z1-Offspring.obj),T,1)./W(P,:),[],2);
                case 12
                    % AASF wrong 
                    % 这个sum不太会写，但是如果这样写判断条件改成Population(P(g_old<=g_new)) = Offspring;还能跑，很奇怪
                    % 也还是不对，这个在图上的形状是3-3-3，正常是1-1-1
                    g_old = max(Population(P).objs-repmat(Z1,T,1)./W(P,:),[],2);%+10e-6*sum((Population(P).objs-repmat(Z1,T,1))./W(P,:),2);
                    g_new = max(repmat((Offspring.obj-Z1),T,1)./W(P,:),[],2);%+10e-6*sum((Offspring.obj-Z1)./W(P,:),2);
                case 13
                    % 单项式PBI ok
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(Offspring.obj-Z,T,1).* W(P,:),2)./normW./normO;
                    g_old   = normP.*CosineP.*normP.*(sqrt(1-CosineP.^2).^(Global.M-1));
                    g_new   = normO.*CosineO.*normO.*(sqrt(1-CosineO.^2).^(Global.M-1));        
                case 14
                    % Local decomposition with shared region
                    % Local PBI
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                    angle1  = acos((W(P,:).*(Offspring.obj-Z))/(norm(W(P,:))*(normO))); %wrong
                    %angle   = sum(sum(mink(angle1,Global.M)));
                    g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                    if  angle1 <= angle  %wrong
                        g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2); 
                    else
                        g_new   = ones(size(g_old,1),size(g_old,2)).*inf;
                        
                    end
                    %end

                    
                    
                    
                case 16
                    % APD in RVEA
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    angle1  = acos(dot(W(P,:),(Population(P).objs-repmat(Z,T,1)))./(norm(W(P,:))*(normP)));
                    %angle11 = acos(dot(W(P,:),(Offspring.obj-Z))./(norm(W(P,:))*(normO)));
                    angle11 = acos(dot(W(P,:),(Population(P).objs-repmat(Z,T,1)))./(norm(W(P,:))*(normO)));%wrong
                    %angle1  = dot(W(P,:),W(P,:)).*(normP);
                    angle2  = max(acos(dot(W(P,:),W(P,:))/(norm(W(P,:))*norm(W(P,:)))),[],2);%wrong
                    publish_alog1 = Global.M.*(Global.gen/Global.maxgen).*(angle1/angle2);
                    
                    
                    publish_alog2 = Global.M.*(Global.gen/Global.maxgen).*(angle11/angle2);
                    g_old   = (1 + publish_alog1).*normP;
                    g_new   = (1 + publish_alog2).*normO;
                    %g_old   = max((1 + publish_alog).*normP,[],2);
                    %g_new   = max((1 + publish_alog).*normO,[],2);
                case 17
                    % Knee Decompositon 把g_new
                    % Population换成offspring才对好像，但是跑不对
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                    g_old   = sum(sum(Population(P).objs*(1/sqrt(Global.M))))-(1/sqrt(Global.M))+normP.*sqrt(1-CosineP.^2);
                    %g_new   = sum(sum(Offspring.obj*(1/sqrt(Global.M))))-(1/sqrt(Global.M))+normO.*sqrt(1-CosineO.^2);
                    g_new   = sum(sum(Population(P).objs*(1/sqrt(Global.M))))-(1/sqrt(Global.M))+normO.*sqrt(1-CosineO.^2);
                    
                    
            end
            Population(P(g_old>=g_new)) = Offspring;
            %Population(P(g_old1<=g_new1)) = Offspring;%case 10 11
            %Population(P(g_old<=g_new)) = Offspring; %case12
        end
    end
end