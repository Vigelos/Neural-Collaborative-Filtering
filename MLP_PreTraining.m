function MLP_net = MLP_PreTraining(training_data,User,Item,batch_size,epoch)

Item = Item';
train_data = zeros(batch_size,64);
train_result = [true(batch_size/2,1);false(batch_size/2,1)];

MLP_net = feedforwardnet([32,16]);
MLP_net.layers{1}.transferfcn = 'poslin';
MLP_net.layers{2}.transferfcn = 'poslin';
MLP_net.layers{3}.transferfcn = 'poslin';
MLP_net.trainParam.showWindow = false;


% simple positive and negative
r = zeros(batch_size,1);
c = zeros(batch_size,1);

for iteration=1:epoch
    positive = find(training_data==1);
    temp = randperm(size(positive,1));
    positive = positive(temp(1:batch_size/2));
    r(1:batch_size/2) = mod(positive,size(training_data,1));
    c(1:batch_size/2) = (positive-r(1:batch_size/2))/size(training_data,1);

    negative = find(training_data==0);
    temp = randperm(size(negative,1));
    negative = negative(temp(1:batch_size/2));
    r(batch_size/2+1:end) = mod(negative,size(training_data,1));
    c(batch_size/2+1:end) = (negative-r(batch_size/2+1:end))/size(training_data,1);
    r = r+1;
    c = c+1;

    for i=1:batch_size
        train_data(i,:) = [User(r(i),:),Item(c(i),:)];
    end
    train_data = mapminmax(train_data);

    % train the GFM network
    MLP_net = train(MLP_net,train_data',train_result'); 
end


end

