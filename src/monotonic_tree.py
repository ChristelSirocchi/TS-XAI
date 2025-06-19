import numpy as np

class MonotonicDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        p = np.mean(y)
        return 1 - p**2 - (1 - p)**2

    def _split(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            return {'leaf': True, 'value': np.mean(y)}

        best_feat, best_thresh, best_score = None, None, float('inf')
        best_left, best_right = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx

                if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                    continue

                y_left, y_right = y[left_idx], y[right_idx]

                # Enforce monotonicity
                if np.mean(y_left) > np.mean(y_right):
                    continue

                gini = (len(y_left) * self._gini(y_left) + len(y_right) * self._gini(y_right)) / len(y)

                if gini < best_score:
                    best_feat = feature
                    best_thresh = threshold
                    best_score = gini
                    best_left = (X[left_idx], y[left_idx])
                    best_right = (X[right_idx], y[right_idx])

        if best_feat is None:
            return {'leaf': True, 'value': np.mean(y)}

        return {
            'leaf': False,
            'feature': best_feat,
            'threshold': best_thresh,
            'left': self._split(*best_left, depth + 1),
            'right': self._split(*best_right, depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._split(np.array(X), np.array(y), 0)

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in np.array(X)])

    def print_rules(self, feature_names=None):
        def recurse(node, depth=0, conditions=[]):
            indent = "  " * depth
            if node['leaf']:
                rule = " and ".join(conditions) if conditions else "True"
                predicted_class = int(np.argmax(node['value']))
                print(f"{indent}if {rule}: predict class {predicted_class}")
                return
            feat = node['feature']
            thresh = node['threshold']
            feat_name = feature_names[feat] if feature_names is not None else f"X[{feat}]"
    
            # Left
            recurse(node['left'], depth + 1, conditions + [f"{feat_name} <= {thresh:.3f}"])
            # Right
            recurse(node['right'], depth + 1, conditions + [f"{feat_name} > {thresh:.3f}"])
    
        recurse(self.tree)
