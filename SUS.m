function  [H_sus,S] = SUS(H,U_s,a,antenna)
K= length(H(:,1));%%전체 user수
%U_s 선택해야하는 user수
%H is (number of user X number of antenna) matrix
% step 1. initialization:
T = 1:K;
H_sus = [];
g = [];
i = 1;
Tlen = [];
S=[];
M=antenna;%transmit anntena 수
% USER SELECTION
while(i<=U_s && isempty(T)~=1)
    
    %     fprintf('SUS number %d\n',length(T));
    
    Tlen(i) = length(T);
    f = zeros(length(T),M);          %  g_k
    f_norm = zeros(1,length(T));     %  ||g_k||
    for k = 1:length(T)
        
        % Find User in T with max norm orthogonal to span{g}
        temp = zeros(M);
        for j=1:i-1
            temp = temp + (g(j,:)'*g(j,:))/norm(g(j,:))^2;
        end
        
        f(k,:) = H(T(k),:)*(eye(M)-temp) ;
        f_norm(k) = norm(f(k,:));
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%
    [~,sel] = max(f_norm);
    
    H_sus = [H_sus; H(T(sel),:)];
    g = [g; f(sel,:)];
    S=[S T(sel)];
    
    % Update T
    T(sel) = [];%선택된 sel 번째 user를 삭제
    temp = [];
    for k = 1:length(T)
        
        if abs(H(T(k),:)*g(i,:)')/(norm(H(T(k),:))*norm(g(i,:))) < a
            temp = [temp T(k)];
        end
        
    end
    T = temp;
    
    i = i+1; % Next Iteration
    
end

if U_s==K
    H_sus=H;
    S=1:K;
    
end

%%%%%%%
S=sort(S);
for k=1:length(S)
    H_sus(k,:)=H(S(k),:);
end
%%%%%%%%

end