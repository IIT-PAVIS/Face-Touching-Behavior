%%

clc
clc
clear all
clear all

%convert json to mat and save it as mat under mat_json_openPose
%You need to download the code in: https://it.mathworks.com/matlabcentral/fileexchange/20565-json-parser
%to use the parse_json function that is being called by convertjsonToMat.m

v_json = 'C:\Users\cbeyan\Desktop\faceTouching\leadershipDetection\jsons_openPose\'; %openPose results in jsons format
v_mat = 'C:\Users\cbeyan\Desktop\faceTouching\leadershipDetection\mat_json_openPose\'; %directory to save the openpose results as mat files
jsonFiles_all=dir(v_json);
jsonFiles_all(1)=[];
jsonFiles_all(1)=[];

for i=1:size(jsonFiles_all,1) %13 sorunlu
    convertjsonToMat(v_json, [jsonFiles_all(i).name],[v_mat jsonFiles_all(i).name(1:end-5) '.mat']);
end

%%
clearex('v_mat')
mat_all=dir(v_mat);
mat_all(1)=[];
mat_all(1)=[];

for klm=1:size(mat_all,1)
    load([v_mat mat_all(klm).name]); %load a file called data
    my_prediction=[];
    %tic
    for i=1:size(data,2)
        if size(data{1,i}.faces,1)>0 % person detection
            curr_rec(1,1)=data{1,i}.faces{1,1}.face_rectangle{1}-data{1,i}.faces{1,1}.face_rectangle{3}; %x of top-left
            curr_rec(1,2)=data{1,i}.faces{1,1}.face_rectangle{2}-data{1,i}.faces{1,1}.face_rectangle{4}; %y of top-left
            curr_rec(1,3)=data{1,i}.faces{1,1}.face_rectangle{3}*2; % width
            curr_rec(1,4)=data{1,i}.faces{1,1}.face_rectangle{4}*2; %height
    
            list_single_handJoint_x=[];
            list_single_handJoint_y=[];
            for ll=1:size(data{1,i}.right_hands{1,1},2)
                single_handJoint(1,1)=data{1,i}.right_hands{1,1}{1,ll}{1}; %right hand joint 3, [x y prob]. y value
                single_handJoint(1,2)=data{1,i}.right_hands{1,1}{1,ll}{2}; %right hand joint 3, [x y prob]. y value
                right_list_single_handJoint_x(ll)=single_handJoint(1,1);
                right_list_single_handJoint_y(ll)=single_handJoint(1,2);
            end

            bbox_right_hand=[min(right_list_single_handJoint_x), min(right_list_single_handJoint_y),(max(right_list_single_handJoint_x)-min(right_list_single_handJoint_x)),(max(right_list_single_handJoint_y)-min(right_list_single_handJoint_y))];

            for ll=1:size(data{1,i}.left_hands{1,1},2)
                single_handJoint(1,1)=data{1,i}.left_hands{1,1}{1,ll}{1}; %right hand joint 3, [x y prob]. y value
                single_handJoint(1,2)=data{1,i}.left_hands{1,1}{1,ll}{2}; %right hand joint 3, [x y prob]. y value
                left_list_single_handJoint_x(ll)=single_handJoint(1,1);
                left_list_single_handJoint_y(ll)=single_handJoint(1,2);
            end
            bbox_left_hand=[min(left_list_single_handJoint_x), min(left_list_single_handJoint_y),(max(left_list_single_handJoint_x)-min(left_list_single_handJoint_x)),(max(left_list_single_handJoint_y)-min(left_list_single_handJoint_y))];

            area_left_hand= ((max(left_list_single_handJoint_x)-min(left_list_single_handJoint_x)))*((max(left_list_single_handJoint_y)-min(left_list_single_handJoint_y)));
            area_right_hand= ((max(right_list_single_handJoint_x)-min(right_list_single_handJoint_x)))*((max(right_list_single_handJoint_y)-min(right_list_single_handJoint_y)));
    
            center_left_hand(1,1)=(min(left_list_single_handJoint_x))+((max(left_list_single_handJoint_x)-min(left_list_single_handJoint_x))/2);
            center_left_hand(1,2)=(min(left_list_single_handJoint_y))+((max(left_list_single_handJoint_y)-min(left_list_single_handJoint_y))/2);
            center_right_hand(1,1)=(min(right_list_single_handJoint_x))+((max(right_list_single_handJoint_x)-min(right_list_single_handJoint_x))/2);
            center_right_hand(1,2)=(min(right_list_single_handJoint_y))+((max(right_list_single_handJoint_y)-min(right_list_single_handJoint_y))/2);

            center_face=[data{1,i}.faces{1,1}.face_rectangle{1} data{1,i}.faces{1,1}.face_rectangle{2}];
            area_face=curr_rec(1,3)*curr_rec(1,4);
    
            if bbox_left_hand(1,1)==0 && bbox_left_hand(1,1)==0 && bbox_left_hand(1,1)==0 && bbox_left_hand(1,1)==0
                bbox_left_hand=[1 1 1 1];
            end
            overlapRatio_leftHandFace = bboxOverlapRatio(bbox_left_hand, curr_rec);
            if bbox_right_hand(1,1)==0 && bbox_right_hand(1,1)==0 && bbox_right_hand(1,1)==0 && bbox_right_hand(1,1)==0
                bbox_right_hand=[1 1 1 1];
            end
            overlapRatio_rightHandFace = bboxOverlapRatio(bbox_right_hand, curr_rec);
    
            if overlapRatio_leftHandFace==0 && overlapRatio_rightHandFace==0
                % title('NO TOUCH')
                my_prediction(i,1)=0;
            elseif (overlapRatio_leftHandFace>0 && area_face<area_left_hand) || (overlapRatio_rightHandFace>0 && area_face<area_right_hand)
                % title('NO TOUCH')
                my_prediction(i,1)=0;
            else
                % title('TOUCH')
                my_prediction(i,1)=1;
            end
            %toc
       
        else
            my_prediction(i,1)=0;  
        end
    end
    save(['prediction_method1\' mat_all(klm).name], 'my_prediction')
    clearex('data', 'i', 'v_mat','mat_all')
end % end of reading mat files