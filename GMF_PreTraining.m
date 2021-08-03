function h = GMF_PreTraining(training_data,User,Item,batch_size,epoch)
% get the core for the GMF part
learning_rate_negative = 0.01;
learning_rate_positive = 0.012;

Item = Item';

h = ones(8,1)+(rand(8,1)-0.5);

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

    for i=1:batch_size/2 %positive
        s = User(r(i),:).*Item(c(i),:)*h;
        y=1;
        prefix = 1/(1+exp(-s));
        h = h-learning_rate_positive*(prefix-y)*h;
    end
    for i=batch_size/2+1:batch_size %negative
        s = User(r(i),:).*Item(c(i),:)*h;
        y=0;
        prefix = 1/(1+exp(-s));
        h = h-learning_rate_negative*(prefix-y)*h;
    end

end

end
