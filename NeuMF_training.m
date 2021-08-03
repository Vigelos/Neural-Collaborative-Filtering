function [GMF_h,MLP_h] = NeuMF_training(GMF_h,MLP_net,MLP_h,alpha,interactionMatrix_train,...
                                        User_MLP,Item_MLP,User_GMF,Item_GMF,...
                                        batch_size,epoch)
%NEUMF_TRAINING use the pre-trained GMF and MLP, train the H in the intergrated model
learning_rate = 0.01;

%construct data set
data = zeros(size(interactionMatrix_train,1)*size(interactionMatrix_train,2),3);
for i=1:size(interactionMatrix_train,1)
    for j=1:size(interactionMatrix_train,2)
        data((i-1)*size(interactionMatrix_train,2)+j,1) = i;
        data((i-1)*size(interactionMatrix_train,2)+j,2) = j;
        data((i-1)*size(interactionMatrix_train,2)+j,3) = interactionMatrix_train(i,j);
    end
end

last_valid = 10000;
for epoch_iteration = 1:epoch
    loss = 0;
    [train_data,valid_data,~] = dividerand(data',0.8,0.2,0);  
    
    for batch_iteration = 1:batch_size %training
        select = train_data(:,randperm(1024,1));
        GMF_core = (User_GMF(select(1),:)').*Item_GMF(:,select(2));
        MLP_core = MLP_net([User_MLP(select(1),:)';Item_MLP(:,select(2))]);
        
        s = alpha*sum(GMF_core.*GMF_h)+(1-alpha)*sum(MLP_core.*MLP_h);
        
        GMF_h = GMF_h+learning_rate*(1/(1+exp(-s))-select(3))*alpha*GMF_h ;
        MLP_h = MLP_h+learning_rate*(1/(1+exp(-s))-select(3))*(1-alpha)*MLP_h;
    end
    
    for batch_iteration = 1:batch_size % validation
        if size(valid_data,2)>batch_size
            valid_size = batch_size;
        else
            valid_size = size(valid_data,2);
        end
        select = valid_data(:,randperm(valid_size,1));
        GMF_core = (User_GMF(select(1),:)').*Item_GMF(:,select(2));
        MLP_core = MLP_net([User_MLP(select(1),:)';Item_MLP(:,select(2))]);
        s = alpha*sum(GMF_core.*GMF_h)+(1-alpha)*sum(MLP_core.*MLP_h);
        p = 1/(1+exp(-s));
        l = -(select(3)*log(p) + (1-select(3))*log(1-p));
        loss = loss+l;
    end
    
    fprintf('valid loss in epoch %d : %d\n',epoch_iteration,loss/valid_size); 
   
    if loss/valid_size<last_valid
        last_valid = loss/valid_size;
    else
        break
    end
end

end

