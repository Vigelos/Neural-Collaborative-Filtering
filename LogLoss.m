function loss = LogLoss(User_MLP,Item_MLP,User_GMF,Item_GMF,GMF_h,MLP_h,MLP_net,interactionMatrix_test,alpha)

%construct data set
data = zeros(size(interactionMatrix_test,1)*size(interactionMatrix_test,2),3);
for i=1:size(interactionMatrix_test,1)
    for j=1:size(interactionMatrix_test,2)
        data((i-1)*size(interactionMatrix_test,2)+j,1) = i;
        data((i-1)*size(interactionMatrix_test,2)+j,2) = j;
        data((i-1)*size(interactionMatrix_test,2)+j,3) = interactionMatrix_test(i,j);
    end
end

loss = 0;
select = randperm(size(interactionMatrix_test,1)*size(interactionMatrix_test,2)*0.1);
data = data(select,:);
for i=1:size(data,2)
    GMF_core = (User_GMF(data(i,1),:)').*Item_GMF(:,data(i,2));
    MLP_core = MLP_net([User_MLP(data(i,1),:)';Item_MLP(:,data(i,2))]);
    s = alpha*sum(GMF_core.*GMF_h)+(1-alpha)*sum(MLP_core.*MLP_h);
    p = 1/(1+exp(-s));
    l = -(data(i,3)*log(p) + (1-data(i,3 ))*log(1-p));
    loss = loss+l;
end
loss = loss/i;

end