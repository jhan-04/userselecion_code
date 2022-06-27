function [predicted_labels,accuracy] = KNN_classifier(k,data_train,label_train,data_test,label_test)

%data_train 1 byfeature วเบคลอ
ed=zeros(length(label_train),length(label_test));

for test=1:length(label_test)
    
    for train=1:length(label_train)
        
        dist(test,train)=sum((data_train(train,:)-data_test(test,:)).^2);
        
        
    end
    [distance(test,:),index(test,:)]= sort(dist(test,:));
    
end


%find the nearest k for each data point of the testing data
k_nn_ind=index(:,1:k);


%get the majority vote
for test=1:length(label_test)
    
    K_label(test,:)=label_train(index(test,1:k));

end
predicted_labels=(mode(K_label,2))';


%calculate the classification accuracy
if isempty(label_test)==0
    accuracy=length(find(predicted_labels==label_test))/length(label_test);
end




end