function [clusters, centroids] = k_means(X, k)

[m, n] = size(X);
max_iters = 100;
tolerance = 1e-6;

if (k <= 0 || k > m)
  error('k_means:invalidK', 'k must satisfy 1 <= k <= size(X, 1).');
end

perm = randperm(m);
centroids = X(perm(1:k), :);
clusters = zeros(m, 1);
can_draw = (n == 2) && ~isempty(available_graphics_toolkits());

for iter = 1:max_iters
  distances = zeros(m, k);
  for cluster_idx = 1:k
    diff = X - centroids(cluster_idx, :);
    distances(:, cluster_idx) = sum(diff .^ 2, 2);
  end

  [~, new_clusters] = min(distances, [], 2);
  new_centroids = centroids;

  for cluster_idx = 1:k
    members = (new_clusters == cluster_idx);
    if any(members)
      new_centroids(cluster_idx, :) = mean(X(members, :), 1);
    else
      new_centroids(cluster_idx, :) = X(randi(m), :);
    end
  end

  if (can_draw)
    draw_clusters(X, new_clusters, new_centroids);
    title(sprintf('K-means: k = %d, iteration = %d', k, iter));
    drawnow;
  end

  centroid_shift = max(abs(new_centroids(:) - centroids(:)));
  if isequal(new_clusters, clusters) && centroid_shift < tolerance
    clusters = new_clusters;
    centroids = new_centroids;
    break;
  end

  clusters = new_clusters;
  centroids = new_centroids;
end
end
