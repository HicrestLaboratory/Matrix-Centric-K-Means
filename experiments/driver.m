

function driver(file_path, n, d, k, n_trials)

    disp("Reading libsvm file...");
    X = read_libsvm(file_path, n, d);
    disp("Done!");

    fprintf("n:%d d:%d k:%d\n", n, d, k);


    times = zeros(1, n_trials);
    scores = zeros(1, n_trials);

    for i=1:n_trials

        tic;
        score = knKmeans(X, k);
        etime = toc;

        times(i) = etime;
        scores(i) = score;
        
        disp(score)
        disp(etime)
    end
    disp(times)

    time_kmeans = mean(times(2:end));
    score_kmeans = mean(scores(2:end));

    fprintf("Time: %.8f\n", time_kmeans);
    fprintf("Score: %.8f\n", score_kmeans);
end
