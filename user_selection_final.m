clear;clc;

tic;

Nt=4;%U=유저수, Nt=안테나수,
U_s=2;% 선택할 유저수


P=10^(5/10)/1000;%signal power 5dBm
N0=10^(-80/10)/1000;%noise power-80dBm

z=1;

for U=3:1:10
    U;
    

    
    
    N=1:1:U; % N=user index
    S=nchoosek(N,U_s); %N개에서 U_s개 고르는 경우의 수 조합 
    SS=length(S(:,1));%N개에서 U_s개 고르는 경우의 수 indx
    
    train=20000;
    test=2000;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%train
    data_train=zeros(train,U*(U+1)/2);
    Rsum=zeros(1,SS);
    for m=1:1:train
        
        H=randn(U,Nt)+randn(U,Nt)*1i;%채널
        
        for j=1:SS
            for k=1:U_s
                % HH(k,:,SS)=H(S(j,k),:)
                HH(k,:)=H(S(j,k),:);
            end
            w=HH'*inv(HH*HH');
            w_=w/norm(w);
            Rsum(j)=0;
            for k=1:U_s
                Rsum(j)=Rsum(j)+log2(1+P*abs(HH(k,:)*w_(:,k)).^2/N0);
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%
        feat=[];
        for user=1:1:U
            % feat=[ feat norm(H(user,:))];
            for other=user:1:U
                feat=[ feat abs(H(user,:)* H(other,:)')];
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%
        data_train(m,:)=feat;
        label_train(m)=find(Rsum==max(Rsum)); %가장 좋은 라벨
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%test
    
    data_test=zeros(test,U*(U+1)/2);
    H_test=zeros(U,Nt,test);
    Rsum=zeros(1,SS);
    for n=1:1:test
        
        H=randn(U,Nt)+randn(U,Nt)*1i;%채널
        for j=1:SS
            for k=1:U_s
                HH(k,:)=H(S(j,k),:);
            end
            w=HH'*inv(HH*HH');
            w_=w/norm(w);
            Rsum(j)=0;
            for k=1:U_s
                Rsum(j)=Rsum(j)+log2(1+P*abs(HH(k,:)*w_(:,k)).^2/N0);
            end
        end
        %%%%%%%%%%%%%%%%%%%%%
        feat=[];
        for user=1:1:U
            for other=user:1:U
                feat=[ feat abs(H(user,:)* H(other,:)')];
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%
        
        data_test(n,:)=feat;
        label_test(n)=find(Rsum==max(Rsum)); %가장 좋은 라벨
        M(n)=max(Rsum); %Rsum의 Maximum값=optimal
        
        H_test(:,:,n)=H;
        
        
    end
    
    
    
    [predicted_labels_knn,accuracy_knn(z)] = KNN_classifier(233,data_train,label_train,data_test,label_test);
    [predicted_labels_svm,accuracy_svm(z)] = SVM_classifier(data_train,label_train,data_test,label_test);
    
    
    %%%%%%%%%%svm rate
    for n=1:1:test
        H=H_test(:,:,n);
        
        j=predicted_labels_svm(n);
        for k=1:U_s
            HH(k,:)=H(S(j,k),:);
        end
        w=HH'*inv(HH*HH');
        w_=w/norm(w);
        Rsum_svm(n)=0;
        for k=1:U_s
            Rsum_svm(n)=Rsum_svm(n)+log2(1+P*abs(HH(k,:)*w_(:,k)).^2/N0);
        end
    end
    
    
        %%%%%%%%%%knn rate
    for n=1:1:test
        H=H_test(:,:,n);
        
        j=predicted_labels_knn(n);
        for k=1:U_s
            HH(k,:)=H(S(j,k),:);
        end
        w=HH'*inv(HH*HH');
        w_=w/norm(w);
        Rsum_knn(n)=0;
        for k=1:U_s
            Rsum_knn(n)=Rsum_knn(n)+log2(1+P*abs(HH(k,:)*w_(:,k)).^2/N0);
        end
    end
    
    
    
    
    
    %%%%%%%%%%SUS1
    a=0.3;
    acc_sus1=0;
    for n=1:1:test
        H=H_test(:,:,n);
        [H_sus,S_sus] = SUS(H,U_s,a,Nt);
        w=H_sus'*inv(H_sus*H_sus');
        w_=w/norm(w);
        Rsum_sus1(n)=0;
        for k=1:length(S_sus)
            Rsum_sus1(n)=Rsum_sus1(n)+log2(1+P*abs(H_sus(k,:)*w_(:,k)).^2/N0);
        end
        same=(sum(S_sus==S(label_test(n),:)))==2;
        acc_sus1=acc_sus1+same;
        acc_sus1;
    end
    acc_sus_1(z)=acc_sus1/test;
    
    %%%%%%%%%%SUS2
    a=0.6;
    acc_sus2=0;
    for n=1:1:test
        H=H_test(:,:,n);
        [H_sus,S_sus] = SUS(H,U_s,a,Nt);
        w=H_sus'*inv(H_sus*H_sus');
        w_=w/norm(w);
        Rsum_sus2(n)=0;
        for k=1:length(S_sus)
            Rsum_sus2(n)=Rsum_sus2(n)+log2(1+P*abs(H_sus(k,:)*w_(:,k)).^2/N0);
        end
        same=(sum(S_sus==S(label_test(n),:)))==2;
        acc_sus2=acc_sus2+same;
        acc_sus2;
    end
    acc_sus_2(z)=acc_sus2/test;
    
    
    optlmal(z)=mean(M);
    svm_opt(z)=mean(Rsum_svm);
    sus1(z)=mean(Rsum_sus1);
    sus2(z)=mean(Rsum_sus2);
    knn_opt(z)=mean(Rsum_knn);
    z=z+1;
end

% figure(1)
% 
% U=3:1:10;
% plot(U,optlmal,'r.-',U,svm_opt,'b:',U,knn_opt,'m',U,sus1,'k.-',U,sus2,'c--');
% xlabel('number of users U');
% ylabel('sum of throughput of selected users');
% legend('optimal','SVM','KNN','SUS a=0.3','SUS a=0.6');


figure(1)
U=3:1:10;
plot(U,optlmal,'r.-',U,svm_opt,'b',U,knn_opt,'m:',U,sus2,'c--');
xlabel('number of users U');
ylabel('sum of throughput of selected users');
legend('optimal','SVM','KNN','SUS a=0.6')

figure(3)
plot(U,accuracy_svm,'b',U,accuracy_knn,'m',U,acc_sus_1,'k.-',U,acc_sus_2,'c--');
xlabel('number of users U');
ylabel('accuracy');
legend('SVM','KNN','SUS a=0.4','SUS a=0.6');

toc;