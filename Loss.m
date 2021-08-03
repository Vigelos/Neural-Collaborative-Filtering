function loss = Loss(User,Item,interactionMatrix_test)
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
select = randperm(size(interactionMatrix_test,1)*size(interactionMatrix_test,2)*0.3);
data = data(select,:);
for i=1:size(data,1)
    s = User(data(i,1),:)*Item(:,data(i,2));
    p = 1/(1+exp(-s));
    l = -(data(i,3)*log(p) + (1-data(i,3 ))*log(1-p));
    loss = loss+l;
end
loss = loss/i;

end