function [vd,vq] = GWO(i_ref,idq,omega)
    persistent v_old Ls Rs psi_s Ts Vdc idmax iqmax radius

    if isempty(v_old) 
        v_old = zeros(2,1); %[-1.1552; 9.6879];
        Ls = 0.32e-3; % H
        Rs = 0.285; % ohm
        psi_s = 0.0079; % Vs

        % Sampling time
        Ts = 1e-4; % s
    
        % DC voltage
        Vdc = 24; % V

        idmax = 1;
        iqmax = 5;

        radius = Vdc/sqrt(3);
    end

    vd = v_old(1);
    vq = v_old(2);

    iprev = [idq(1); idq(2)]; % measured currents
    
    % Prediction
    % x[k+1] = A*x[k]+B*u[k]+d
    % matrices
    A = [1-Ts*Rs/Ls, omega*Ts; -omega*Ts, 1-Ts*Rs/Ls];
    B = Ts/Ls*eye(2);
    d = [0 -omega*Ts*psi_s/Ls]';

    % k+1
    ik_p1 = A*iprev+B*[v_old(1);v_old(2)]+d;


    Max_iter = 20;    % max iteration 
    popSize = 10;     % population size
    
    % Inicializálás
    Position = generatePoints(radius,popSize);  % init pop
    Alpha_pos = zeros(2, 1);                    % Alpha position
    Beta_pos = zeros(2,1);                      % Beta pos
    Delta_pos = zeros(2,1);                     % Delta pos
    Alpha_score = inf;                          % Alpha score (minimal value)
    Beta_score = inf;                           % Beta score
    Delta_score = inf;                          % Delta score
    
    % main loop
    for t = 1:Max_iter
        for i = 1:popSize
            % Compute fitness
            fitness = CostFunction(Position(:,i),ik_p1,i_ref,A,B,d,idmax,iqmax);
            
            % Update Alpha, Beta and Delta pos
            if fitness < Alpha_score
                Alpha_score = fitness;   
                Alpha_pos = Position(:,i); 
            elseif fitness < Beta_score
                Beta_score = fitness;   
                Beta_pos = Position(:,i); 
            elseif fitness < Delta_score
                Delta_score = fitness;  
                Delta_pos = Position(:,i); 
            end
        end
        
        % Computation the direction of motion
        a = 2 - t * (2 / Max_iter);         % 'a' linear decreasing
        r1 = rand(2,popSize);                 
        r2 = rand(2,popSize);                  
        
        
        A_coeff = 2 * a * r1 - a;                % A coefficient
        C_coeff = 2 * r2;                         % C coefficient
        D_alpha = abs(C_coeff .* Alpha_pos - Position);   % Distance from alpha
        D_beta = abs(C_coeff .* Beta_pos - Position);     % distance from beta
        D_delta = abs(C_coeff .* Delta_pos - Position);   %distance from delta
    
        % Update position
        %Position = Position - A_coeff .* (D_alpha + D_beta + D_delta);  
        D_mean = (D_alpha + D_beta + D_delta) / 3;
        Position = Position - A_coeff .* D_mean;

        
        % bound check
        for i = 1:popSize
            if norm(Position(:,i)) > radius
                    Position(:,i) =  Position(:,i)./norm(Position(:,i))*radius;
            end
        end

    end

    v_old = Alpha_pos;

end

function points = generatePoints(R,numPoints)
    xp = (2 * rand(2,numPoints) - ones(2,numPoints)) * R; 
    for i=1:numPoints
        while (norm(xp(:,i)) > R)
            xp(:,i) = (2 * rand(2,1) - ones(2,1)) * R; 
        end
    end
    points = xp;
end

function g = CostFunction(v,ik1,iref,A,B,d,idmax,iqmax)
    idmax = 0.5;
    iqmax = 4;
    ik2 = A*ik1+B*v+d;
    g = (iref-ik2(2,:)).^2+ik2(1,:).^2+ 10000 * (abs(ik2(2,:)) > iqmax) + 10000* (abs(ik2(1,:)) > idmax);
end
