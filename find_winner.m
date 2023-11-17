function [winner_vector, winner_idx, winner_distance] = find_winner(data, x)

%
% This function enables to find the closest vector of one given vector
% data : matrix which contains the vectors
% x : vector
% Returns the winner vector, its index and the corresponding distance between it and x
%

% Initialize the winner vector and the corresponding distance from the
% input x

% %------------------------Euclidean norm distance ---------------
% %--------------------------------------------------------
% 
% winner_vector = data(:,1);
% winner_distance = norm(x-winner_vector);
% winner_idx = 1;
% 
% % Loop on the number of vectors
% for j = 1:length(data(1,:));
%     % Calculate the distance
%     distance = norm(x - data(:,j));
%     % Update the winner neuron if the distance found is shorter than
%     % the previous one
%     if (distance <= winner_distance)
%         winner_distance = distance;
%         winner_vector = data(:,j);
%         winner_idx = j;
%     end
% end
% 
% end
% %-------------------------Sequare distance---------------
% %--------------------------------------------------------
% winner_vector = data(:,1);
% winner_distance1 = (x-winner_vector).^2;
% winner_distance2 = (x+winner_vector);
% winner_distance3 =winner_distance1./winner_distance2;
% winner_distance=sum(winner_distance3);
% winner_idx = 1;
% 
% % Loop on the number of vectors
% for j = 1:length(data(1,:));
%     % Calculate the distance
%     winner_distance1 = (x-data(:,j)).^2;
%     winner_distance2 = (x+data(:,j));
%     winner_distance3 =winner_distance1./winner_distance2;
%     distance=sum(winner_distance3);
%     
%     % Update the winner neuron if the distance found is shorter than
%     % the previous one
%     if (distance <= winner_distance)
%         winner_distance = distance;
%         winner_vector = data(:,j);
%         winner_idx = j;
%     end
% end
% 
% end
%-------------------------Hassanat distance---------------
%--------------------------------------------------------
winner_vector = data(:,1);
winner_distance1 = 1+min(x-winner_vector);
winner_distance11 = 1+min(x-winner_vector)+abs(+min(x-winner_vector));
winner_distance2 = 1+max(x+winner_vector);
winner_distance22 = 1+max(x+winner_vector)+abs(+min(x-winner_vector));
if min(x-winner_vector)>=0
winner_distance3 = 1-(winner_distance1./winner_distance2); 
else
winner_distance3 =1-(winner_distance11./winner_distance22);
end
winner_distance=sum(winner_distance3);
winner_idx = 1;

% Loop on the number of vectors
for j = 1:length(data(1,:));
    % Calculate the distance
    winner_distance1 = 1+min(x-data(:,j));
    winner_distance11 = 1+min(x-data(:,j))+abs(+min(x-data(:,j)));
    winner_distance2 = 1+max(x+data(:,j));
    winner_distance22 = 1+max(x+data(:,j))+abs(+min(x-data(:,j)));
    if min(x-data(:,j))>=0
      winner_distance3 = 1-(winner_distance1./winner_distance2); 
    else
      winner_distance3 =1-(winner_distance11./winner_distance22);
    end
    distance=sum(winner_distance3);
    
    % Update the winner neuron if the distance found is shorter than
    % the previous one
    if (distance <= winner_distance)
        winner_distance = distance;
        winner_vector = data(:,j);
        winner_idx = j;
    end
end

end