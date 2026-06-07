function main_fuzzy_ga_navigation()

    %% ============================================================
    %  Mini Project: Intelligent Mobile Robot Navigation
    %  Method: Hybrid Fuzzy Logic + Genetic Algorithm
    %
    %  Grid value:
    %  0 = free space
    %  1 = obstacle / wall
    %
    %  Coordinate format:
    %  [row, column]
    %% ============================================================

    rows = 15;
    cols = 20;

    %% ===================== EASY MAP =============================
    % Easy map: minimal obstacles, simple path to goal

    easyMap = zeros(rows, cols);

    % Add boundary walls
    easyMap(1, :) = 1;
    easyMap(end, :) = 1;
    easyMap(:, 1) = 1;
    easyMap(:, end) = 1;

    % Add small obstacle walls
    easyMap(6, 2:10) = 1;
    easyMap(10, 9:19) = 1;
    easyMap(2:7, 14) = 1;

    % Define start and goal point
    startEasy = [2, 2];
    goalEasy  = [14, 18];

    % Make sure start and goal are free space
    easyMap(startEasy(1), startEasy(2)) = 0;
    easyMap(goalEasy(1), goalEasy(2)) = 0;


    %% ===================== COMPLEX MAP ==========================
    % Complex map: more walls, maze-like structure, but goal is reachable

    complexMap = zeros(rows, cols);

    % Add boundary walls
    complexMap(1, :) = 1;
    complexMap(end, :) = 1;
    complexMap(:, 1) = 1;
    complexMap(:, end) = 1;

    % Add internal maze walls
    complexMap(3, 3:15) = 1;
    complexMap(5, 5:18) = 1;
    complexMap(7, 2:12) = 1;
    complexMap(9, 6:19) = 1;
    complexMap(11, 2:15) = 1;
    complexMap(13, 5:18) = 1;

    % Create openings so robot can pass
    complexMap(3, 7) = 0;
    complexMap(3, 13) = 0;

    complexMap(5, 10) = 0;
    complexMap(5, 17) = 0;

    complexMap(7, 4) = 0;
    complexMap(7, 11) = 0;

    complexMap(9, 8) = 0;
    complexMap(9, 15) = 0;

    complexMap(11, 6) = 0;
    complexMap(11, 14) = 0;

    complexMap(13, 9) = 0;
    complexMap(13, 17) = 0;

    % Add vertical walls for more complexity
    complexMap(2:6, 17) = 1;
    complexMap(8:12, 4) = 1;
    complexMap(4:10, 13) = 1;

    % Create openings in vertical walls
    complexMap(4, 17) = 0;
    complexMap(10, 4) = 0;
    complexMap(6, 13) = 0;

    % Define start and goal point
    startComplex = [2, 2];
    goalComplex  = [14, 18];

    % Make sure start and goal are free space
    complexMap(startComplex(1), startComplex(2)) = 0;
    complexMap(goalComplex(1), goalComplex(2)) = 0;


    %% ===================== DISPLAY ORIGINAL MAPS =================

    figure('Name', 'Original Maps');

    subplot(1, 2, 1);
    plotGridMap(easyMap, startEasy, goalEasy, 'Easy Map');

    subplot(1, 2, 2);
    plotGridMap(complexMap, startComplex, goalComplex, 'Complex Map');


    %% ===================== GA OPTIMIZATION =======================

    % The GA will optimize these fuzzy controller weights:
    % weight(1) = goal direction weight
    % weight(2) = obstacle avoidance weight
    % weight(3) = turning smoothness weight
    % weight(4) = repeated path penalty weight

    populationSize = 40;
    maxGenerations = 35;
    maxSteps = 300;

    lowerBound = [0, 0, 0, 0];
    upperBound = [5, 5, 3, 5];

    maps = {easyMap, complexMap};
    starts = {startEasy, startComplex};
    goals = {goalEasy, goalComplex};

    fprintf('\nOptimizing Fuzzy Controller using Genetic Algorithm...\n');

    bestWeights = simpleGA( ...
        maps, starts, goals, ...
        populationSize, maxGenerations, ...
        lowerBound, upperBound, maxSteps);

    fprintf('\nBest Fuzzy-GA Weights Found:\n');
    fprintf('Goal Direction Weight      = %.3f\n', bestWeights(1));
    fprintf('Obstacle Avoidance Weight  = %.3f\n', bestWeights(2));
    fprintf('Turning Smoothness Weight  = %.3f\n', bestWeights(3));
    fprintf('Repeated Path Penalty      = %.3f\n', bestWeights(4));


    %% ===================== FINAL SIMULATION ======================

    [pathEasy, metricEasy] = simulateFuzzyRobot( ...
        easyMap, startEasy, goalEasy, bestWeights, maxSteps);

    [pathComplex, metricComplex] = simulateFuzzyRobot( ...
        complexMap, startComplex, goalComplex, bestWeights, maxSteps);


    %% ===================== DISPLAY FINAL PATHS ==================

    figure('Name', 'Fuzzy-GA Navigation Result');

    subplot(1, 2, 1);
    plotGridMapWithPath(easyMap, startEasy, goalEasy, pathEasy, ...
        'Easy Map - Fuzzy-GA Path');

    subplot(1, 2, 2);
    plotGridMapWithPath(complexMap, startComplex, goalComplex, pathComplex, ...
        'Complex Map - Fuzzy-GA Path');


    %% ===================== PERFORMANCE RESULT ===================

    fprintf('\n==================== PERFORMANCE RESULT ====================\n');

    fprintf('\nEasy Map Result:\n');
    disp(metricEasy);

    fprintf('\nComplex Map Result:\n');
    disp(metricComplex);
end

%% ============================================================
%                 LOCAL FUNCTIONS (SUPPORTING CODE)
%% ============================================================

function bestWeights = simpleGA(maps, starts, goals, populationSize, maxGenerations, lowerBound, upperBound, maxSteps)

    numberOfWeights = length(lowerBound);

    % Random initial population
    population = zeros(populationSize, numberOfWeights);

    for i = 1:populationSize
        population(i, :) = lowerBound + rand(1, numberOfWeights) .* (upperBound - lowerBound);
    end

    bestFitness = -inf;
    bestWeights = population(1, :);

    for generation = 1:maxGenerations

        fitnessValues = zeros(populationSize, 1);

        for i = 1:populationSize
            fitnessValues(i) = evaluateFitness( ...
                maps, starts, goals, population(i, :), maxSteps);
        end

        % Find best individual
        [currentBestFitness, bestIndex] = max(fitnessValues);

        if currentBestFitness > bestFitness
            bestFitness = currentBestFitness;
            bestWeights = population(bestIndex, :);
        end

        fprintf('Generation %02d | Best Fitness = %.2f\n', generation, bestFitness);

        % Create new population
        newPopulation = zeros(size(population));

        % Elitism: keep the best solution
        newPopulation(1, :) = bestWeights;

        for i = 2:populationSize

            % Selection
            parent1 = tournamentSelection(population, fitnessValues);
            parent2 = tournamentSelection(population, fitnessValues);

            % Crossover
            alpha = rand;
            child = alpha .* parent1 + (1 - alpha) .* parent2;

            % Mutation
            mutationRate = 0.30;

            if rand < mutationRate
                mutationStrength = 0.40;
                child = child + mutationStrength .* randn(1, numberOfWeights);
            end

            % Keep inside limits
            child = max(child, lowerBound);
            child = min(child, upperBound);

            newPopulation(i, :) = child;
        end

        population = newPopulation;
    end
end


function selectedParent = tournamentSelection(population, fitnessValues)

    tournamentSize = 3;
    populationSize = size(population, 1);

    randomIndex = randi(populationSize, tournamentSize, 1);
    [~, bestLocalIndex] = max(fitnessValues(randomIndex));

    selectedParent = population(randomIndex(bestLocalIndex), :);
end


function fitness = evaluateFitness(maps, starts, goals, weights, maxSteps)

    fitness = 0;

    for mapIndex = 1:length(maps)

        currentMap = maps{mapIndex};
        startPoint = starts{mapIndex};
        goalPoint = goals{mapIndex};

        [~, metric] = simulateFuzzyRobot( ...
            currentMap, startPoint, goalPoint, weights, maxSteps);

        if metric.Success == 1
            mapFitness = 1000 ...
                - 3.0 * metric.PathLength ...
                - 1.5 * metric.Turns ...
                - 100 * metric.Collisions;
        end

        if metric.Success == 0
            mapFitness = -500 ...
                - 0.5 * metric.Steps ...
                - 50 * metric.FinalDistance ...
                - 100 * metric.Collisions;
        end

        fitness = fitness + mapFitness;
    end
end


function [path, metric] = simulateFuzzyRobot(map, startPoint, goalPoint, weights, maxSteps)

    % Direction format:
    % 1 = Up
    % 2 = Right
    % 3 = Down
    % 4 = Left

    directions = [
        -1,  0;   % Up
         0,  1;   % Right
         1,  0;   % Down
         0, -1    % Left
    ];

    currentPosition = startPoint;
    currentHeading = 2; % Start facing right

    path = currentPosition;

    visited = zeros(size(map));
    visited(currentPosition(1), currentPosition(2)) = 1;

    collisions = 0;
    invalidMoves = 0;
    turns = 0;

    for step = 1:maxSteps

        if isequal(currentPosition, goalPoint)
            break;
        end

        actionScores = -inf(1, 4);

        for action = 1:4
            actionScores(action) = fuzzyActionScore( ...
                map, currentPosition, goalPoint, ...
                currentHeading, action, weights, visited);
        end

        [~, bestAction] = max(actionScores);

        nextPosition = currentPosition + directions(bestAction, :);

        % Safety check
        if isInvalidMove(map, nextPosition)
            collisions = collisions + 1;
            invalidMoves = invalidMoves + 1;
            break;
        end

        if bestAction ~= currentHeading
            turns = turns + 1;
        end

        currentPosition = nextPosition;
        currentHeading = bestAction;

        visited(currentPosition(1), currentPosition(2)) = ...
            visited(currentPosition(1), currentPosition(2)) + 1;

        path = [path; currentPosition];
    end

    success = isequal(currentPosition, goalPoint);
    stepsTaken = size(path, 1) - 1;
    finalDistance = norm(currentPosition - goalPoint);

    if stepsTaken == 0
        smoothness = 0;
    else
        smoothness = turns / stepsTaken;
    end

    metric = table( ...
        success, ...
        stepsTaken, ...
        stepsTaken, ...
        collisions, ...
        invalidMoves, ...
        turns, ...
        smoothness, ...
        finalDistance, ...
        'VariableNames', { ...
        'Success', ...
        'Steps', ...
        'PathLength', ...
        'Collisions', ...
        'InvalidMoves', ...
        'Turns', ...
        'Smoothness', ...
        'FinalDistance'});
end


function score = fuzzyActionScore(map, currentPosition, goalPoint, currentHeading, action, weights, visited)

    directions = [
        -1,  0;   % Up
         0,  1;   % Right
         1,  0;   % Down
         0, -1    % Left
    ];

    nextPosition = currentPosition + directions(action, :);

    if isInvalidMove(map, nextPosition)
        score = -1e6;
        return;
    end

    %% ================= FUZZY INPUT 1: GOAL PROGRESS ==========
    currentDistance = norm(currentPosition - goalPoint);
    nextDistance = norm(nextPosition - goalPoint);

    progress = currentDistance - nextDistance;

    movingCloser = trapmfManual(progress, [0.05, 0.20, 1.00, 1.10]);
    neutralMove  = trimfManual(progress, [-0.20, 0.00, 0.20]);
    movingAway   = trapmfManual(progress, [-1.10, -1.00, -0.20, -0.05]);

    goalScore = ...
        1.00 * movingCloser + ...
        0.20 * neutralMove  - ...
        0.80 * movingAway;


    %% ============== FUZZY INPUT 2: OBSTACLE CLEARANCE ========
    sensorRange = 5;
    clearance = lookAheadDistance(map, currentPosition, action, sensorRange);

    obstacleNear = trapmfManual(clearance, [-0.10, 0.00, 1.00, 2.00]);
    obstacleMid  = trimfManual(clearance, [1.00, 3.00, 5.00]);
    obstacleFar  = trapmfManual(clearance, [3.00, 5.00, 5.10, 5.20]);

    obstacleScore = ...
        1.00 * obstacleFar + ...
        0.40 * obstacleMid - ...
        1.00 * obstacleNear;


    %% ============== FUZZY INPUT 3: TURNING SMOOTHNESS ========
    turnDifference = mod(action - currentHeading, 4);

    if turnDifference == 0
        turnScore = 1.00;      % straight
    elseif turnDifference == 2
        turnScore = -0.50;     % reverse direction
    else
        turnScore = 0.20;      % left or right turn
    end


    %% ============== REPEATED POSITION PENALTY ================
    repeatedPenalty = visited(nextPosition(1), nextPosition(2));


    %% ============== FINAL FUZZY WEIGHTED SCORE ===============
    score = ...
        weights(1) * goalScore + ...
        weights(2) * obstacleScore + ...
        weights(3) * turnScore - ...
        weights(4) * repeatedPenalty;
end


function distance = lookAheadDistance(map, currentPosition, action, sensorRange)

    directions = [
        -1,  0;   % Up
         0,  1;   % Right
         1,  0;   % Down
         0, -1    % Left
    ];

    distance = 0;

    for k = 1:sensorRange

        checkPosition = currentPosition + k .* directions(action, :);

        if isInvalidMove(map, checkPosition)
            break;
        else
            distance = distance + 1;
        end
    end
end


function invalid = isInvalidMove(map, position)

    row = position(1);
    col = position(2);

    if row < 1 || row > size(map, 1) || col < 1 || col > size(map, 2)
        invalid = true;
        return;
    end

    if map(row, col) == 1
        invalid = true;
    else
        invalid = false;
    end
end


function y = trimfManual(x, params)

    a = params(1);
    b = params(2);
    c = params(3);

    if x <= a || x >= c
        y = 0;
    elseif x == b
        y = 1;
    elseif x > a && x < b
        y = (x - a) / (b - a);
    else
        y = (c - x) / (c - b);
    end
end


function y = trapmfManual(x, params)

    a = params(1);
    b = params(2);
    c = params(3);
    d = params(4);

    if x <= a || x >= d
        y = 0;
    elseif x >= b && x <= c
        y = 1;
    elseif x > a && x < b
        y = (x - a) / (b - a);
    else
        y = (d - x) / (d - c);
    end
end


function plotGridMap(map, startPoint, goalPoint, mapTitle)

    imagesc(map);
    axis equal;
    axis tight;
    grid on;

    title(mapTitle, 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Column');
    ylabel('Row');

    colormap(gca, [1 1 1; 0 0 0]);

    hold on;

    plot(startPoint(2), startPoint(1), 'go', ...
        'MarkerSize', 12, ...
        'MarkerFaceColor', 'g');

    plot(goalPoint(2), goalPoint(1), 'ro', ...
        'MarkerSize', 12, ...
        'MarkerFaceColor', 'r');

    legend('Start', 'Goal', 'Location', 'bestoutside');

    xticks(1:size(map, 2));
    yticks(1:size(map, 1));

    set(gca, 'XGrid', 'on', 'YGrid', 'on', ...
        'GridColor', [0.5 0.5 0.5], ...
        'GridAlpha', 0.5);

    hold off;
end


function plotGridMapWithPath(map, startPoint, goalPoint, path, mapTitle)

    imagesc(map);
    axis equal;
    axis tight;
    grid on;

    title(mapTitle, 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Column');
    ylabel('Row');

    colormap(gca, [1 1 1; 0 0 0]);

    hold on;

    plot(path(:, 2), path(:, 1), 'b-', ...
        'LineWidth', 2);

    plot(path(:, 2), path(:, 1), 'bo', ...
        'MarkerSize', 4, ...
        'MarkerFaceColor', 'b');

    plot(startPoint(2), startPoint(1), 'go', ...
        'MarkerSize', 12, ...
        'MarkerFaceColor', 'g');

    plot(goalPoint(2), goalPoint(1), 'ro', ...
        'MarkerSize', 12, ...
        'MarkerFaceColor', 'r');

    legend('Path', 'Robot Steps', 'Start', 'Goal', 'Location', 'bestoutside');

    xticks(1:size(map, 2));
    yticks(1:size(map, 1));

    set(gca, 'XGrid', 'on', 'YGrid', 'on', ...
        'GridColor', [0.5 0.5 0.5], ...
        'GridAlpha', 0.5);

    hold off;
end