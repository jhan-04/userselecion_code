function [predicted_labels,accuracy_svm] = SVM_classifier(data_train,label_train,data_test,label_test)


%data_train 1 byfeature วเบคลอ
%%OvA

class=unique(label_train);
class_num=length(class);
result = zeros(length(data_test(:,1)),1);
%build models
for j = 1:class_num
    indx = (label_train==class(j)); % Create binary classes for each classifier
  %  SVMModels{j} = fitcsvm(X,indx);
    SVMModels{j}  = fitcsvm(data_train,indx,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
end


%classify test cases

for j=1:size(data_test,1) 
    for k=1:class_num
        [~, score]= predict(SVMModels{k} ,data_test(j,:));
        if length(score)==2
        S(j,k)=score(2);
        else
            S(j,k)=score;
        end
    end
    
A=max(S(j,:),[],2);
predicted_labels(j)=find(A==S(j,:));
end

accuracy_svm=length(find(predicted_labels==label_test))/length(label_test);



end