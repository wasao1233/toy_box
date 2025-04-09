import os
import random
import uuid
import numpy as np
import datetime
import json
import copy
from typing import List, Dict, Any, Optional, Tuple, Callable

from utils.logger import get_logger
from config.config import get_config
from utils.database import get_db
from models.data_models import Model, ModelPerformance

logger = get_logger(__name__)

class GeneticAlgorithm:
    """進化的アルゴリズムクラス
    
    モデルの選択、交配、突然変異を行い、世代を進化させる。
    """
    
    def __init__(self):
        """初期化"""
        self.config = get_config()
        self.db = get_db()
        self.learning_config = self.config.learning
        
        # 乱数シードの設定
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
    
    def initialize_population(
        self,
        model_type: str,
        currency_pair: str,
        hyperparameter_space: Dict[str, Any],
        features: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """初期集団の生成
        
        Args:
            model_type: モデルの種類
            currency_pair: 通貨ペア
            hyperparameter_space: ハイパーパラメータの探索空間
            features: 使用する特徴量のリスト
            
        Returns:
            初期モデル設定のリスト
        """
        population_size = self.learning_config.initial_population
        logger.info(f"初期集団生成: {population_size}個の{model_type}モデルを作成します")
        
        population = []
        
        for i in range(population_size):
            # ランダムなハイパーパラメータを生成
            hyperparameters = self._sample_hyperparameters(hyperparameter_space)
            
            # 特徴量が指定されていない場合は全て使用
            if features is None:
                selected_features = None
            else:
                # 特徴量のサブセットをランダムに選択（最低3つ）
                min_features = min(3, len(features))
                feature_count = random.randint(min_features, len(features))
                selected_features = random.sample(features, feature_count)
            
            # モデル設定を作成
            model_config = {
                "name": f"{model_type}_{currency_pair}_g0_i{i}",
                "model_type": model_type,
                "currency_pair": currency_pair,
                "generation": 0,
                "hyperparameters": hyperparameters,
                "features": selected_features,
                "uuid": str(uuid.uuid4())
            }
            
            population.append(model_config)
        
        return population
    
    def _sample_hyperparameters(self, hyperparameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """ハイパーパラメータの探索空間からランダムにサンプリング
        
        Args:
            hyperparameter_space: ハイパーパラメータの探索空間
                {
                    "param_name": {
                        "type": "int"|"float"|"categorical"|"bool",
                        "min": 最小値 (int/float),
                        "max": 最大値 (int/float),
                        "values": リスト (categorical),
                        "log_scale": 対数スケールかどうか (bool)
                    }
                }
                
        Returns:
            サンプリングされたハイパーパラメータ
        """
        sampled = {}
        
        for param_name, param_config in hyperparameter_space.items():
            param_type = param_config["type"]
            
            if param_type == "int":
                if param_config.get("log_scale", False):
                    # 対数スケールの整数値
                    log_min = np.log10(param_config["min"])
                    log_max = np.log10(param_config["max"])
                    sampled[param_name] = int(10 ** np.random.uniform(log_min, log_max))
                else:
                    # 線形スケールの整数値
                    sampled[param_name] = random.randint(param_config["min"], param_config["max"])
                    
            elif param_type == "float":
                if param_config.get("log_scale", False):
                    # 対数スケールの浮動小数点値
                    log_min = np.log10(param_config["min"])
                    log_max = np.log10(param_config["max"])
                    sampled[param_name] = 10 ** np.random.uniform(log_min, log_max)
                else:
                    # 線形スケールの浮動小数点値
                    sampled[param_name] = np.random.uniform(param_config["min"], param_config["max"])
                    
            elif param_type == "categorical":
                # カテゴリ値
                sampled[param_name] = random.choice(param_config["values"])
                
            elif param_type == "bool":
                # ブール値
                sampled[param_name] = random.choice([True, False])
        
        return sampled
    
    def crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        hyperparameter_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """2つの親モデルから子モデルを生成（交配）
        
        Args:
            parent1: 親モデル1の設定
            parent2: 親モデル2の設定
            hyperparameter_space: ハイパーパラメータの探索空間
            
        Returns:
            子モデルの設定
        """
        # 親のハイパーパラメータをディープコピー
        child_params = {}
        
        # 各ハイパーパラメータについて、親のどちらかから選択または2つの値を混合
        for param_name, param_config in hyperparameter_space.items():
            param_type = param_config["type"]
            
            # ランダムに親の一方を選択
            if random.random() < 0.5:
                # 親1から継承
                child_params[param_name] = parent1["hyperparameters"].get(param_name)
            else:
                # 親2から継承
                child_params[param_name] = parent2["hyperparameters"].get(param_name)
            
            # 数値型のパラメータは確率的に混合
            if param_type in ["int", "float"] and random.random() < 0.3:
                p1_val = parent1["hyperparameters"].get(param_name)
                p2_val = parent2["hyperparameters"].get(param_name)
                
                # 両方の値が存在する場合のみ混合
                if p1_val is not None and p2_val is not None:
                    # 混合比率
                    ratio = np.random.uniform(0, 1)
                    mixed_val = p1_val * ratio + p2_val * (1 - ratio)
                    
                    # 整数型の場合は丸める
                    if param_type == "int":
                        child_params[param_name] = int(round(mixed_val))
                    else:
                        child_params[param_name] = mixed_val
        
        # 特徴量の選択
        if "features" in parent1 and "features" in parent2:
            if parent1["features"] is None or parent2["features"] is None:
                # いずれかがNoneの場合はNoneを継承
                child_features = None
            else:
                # 両方の特徴量を合わせた後、ランダムに選択
                all_features = list(set(parent1["features"] + parent2["features"]))
                
                if len(all_features) <= 3:
                    # 特徴量が少ない場合は全て使用
                    child_features = all_features
                else:
                    # ランダムな数の特徴量を選択
                    feature_count = random.randint(3, len(all_features))
                    child_features = random.sample(all_features, feature_count)
        else:
            child_features = None
        
        # 世代とインデックスの更新
        generation = max(parent1["generation"], parent2["generation"]) + 1
        
        # 子モデルの設定
        child = {
            "name": f"{parent1['model_type']}_{parent1['currency_pair']}_g{generation}_cross",
            "model_type": parent1["model_type"],
            "currency_pair": parent1["currency_pair"],
            "generation": generation,
            "hyperparameters": child_params,
            "features": child_features,
            "uuid": str(uuid.uuid4()),
            "parent_uuid": [parent1["uuid"], parent2["uuid"]]
        }
        
        return child
    
    def mutate(
        self,
        model: Dict[str, Any],
        hyperparameter_space: Dict[str, Any],
        mutation_rate: Optional[float] = None,
        all_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """モデルを突然変異させる
        
        Args:
            model: 元のモデル設定
            hyperparameter_space: ハイパーパラメータの探索空間
            mutation_rate: 突然変異率（指定がなければ設定ファイルの値を使用）
            all_features: 全ての利用可能な特徴量リスト
            
        Returns:
            突然変異後のモデル設定
        """
        if mutation_rate is None:
            mutation_rate = self.learning_config.mutation_rate
        
        # 設定のディープコピー
        mutated = copy.deepcopy(model)
        mutated["uuid"] = str(uuid.uuid4())
        mutated["parent_uuid"] = [model["uuid"]]
        
        # ハイパーパラメータの突然変異
        mutated_params = copy.deepcopy(model["hyperparameters"])
        
        for param_name, param_config in hyperparameter_space.items():
            # 各パラメータを一定確率で変異
            if random.random() < mutation_rate:
                param_type = param_config["type"]
                
                if param_type == "int":
                    if param_config.get("log_scale", False):
                        # 対数スケールの整数値
                        log_min = np.log10(param_config["min"])
                        log_max = np.log10(param_config["max"])
                        mutated_params[param_name] = int(10 ** np.random.uniform(log_min, log_max))
                    else:
                        # 線形スケールの整数値
                        mutated_params[param_name] = random.randint(param_config["min"], param_config["max"])
                        
                elif param_type == "float":
                    if param_config.get("log_scale", False):
                        # 対数スケールの浮動小数点値
                        log_min = np.log10(param_config["min"])
                        log_max = np.log10(param_config["max"])
                        mutated_params[param_name] = 10 ** np.random.uniform(log_min, log_max)
                    else:
                        # 線形スケールの浮動小数点値
                        mutated_params[param_name] = np.random.uniform(param_config["min"], param_config["max"])
                        
                elif param_type == "categorical":
                    # カテゴリ値
                    mutated_params[param_name] = random.choice(param_config["values"])
                    
                elif param_type == "bool":
                    # ブール値
                    mutated_params[param_name] = random.choice([True, False])
        
        mutated["hyperparameters"] = mutated_params
        
        # 特徴量の突然変異
        if all_features and random.random() < mutation_rate:
            if model["features"] is None:
                # 全ての特徴量からランダムに選択
                feature_count = random.randint(3, len(all_features))
                mutated["features"] = random.sample(all_features, feature_count)
            else:
                # 既存の特徴量を変更
                current_features = set(model["features"])
                available_features = set(all_features)
                
                # 追加可能な特徴量
                can_add = list(available_features - current_features)
                
                # 削除可能な特徴量（最低3つは残す）
                can_remove = list(current_features) if len(current_features) > 3 else []
                
                # 追加操作
                if can_add and random.random() < 0.5:
                    to_add = random.choice(can_add)
                    mutated["features"] = list(current_features) + [to_add]
                
                # 削除操作
                elif can_remove and random.random() < 0.5:
                    to_remove = random.choice(can_remove)
                    new_features = list(current_features)
                    new_features.remove(to_remove)
                    mutated["features"] = new_features
        
        # 名前の更新
        mutated["name"] = f"{model['model_type']}_{model['currency_pair']}_g{model['generation']}_mutated"
        
        return mutated
    
    def select_parents(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        n_parents: int,
        method: str = "tournament"
    ) -> List[Dict[str, Any]]:
        """親選択
        
        Args:
            population: モデル集団
            fitness_scores: 各モデルの適合度スコア
            n_parents: 選択する親の数
            method: 選択方法 ("tournament", "roulette")
            
        Returns:
            選択された親モデルのリスト
        """
        if method == "tournament":
            return self._tournament_selection(population, fitness_scores, n_parents)
        elif method == "roulette":
            return self._roulette_selection(population, fitness_scores, n_parents)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _tournament_selection(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        n_parents: int
    ) -> List[Dict[str, Any]]:
        """トーナメント選択
        
        Args:
            population: モデル集団
            fitness_scores: 各モデルの適合度スコア
            n_parents: 選択する親の数
            
        Returns:
            選択された親モデルのリスト
        """
        tournament_size = self.learning_config.tournament_size
        parents = []
        
        for _ in range(n_parents):
            # トーナメント参加者をランダムに選択
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # 最良の個体を選択
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def _roulette_selection(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        n_parents: int
    ) -> List[Dict[str, Any]]:
        """ルーレット選択
        
        Args:
            population: モデル集団
            fitness_scores: 各モデルの適合度スコア
            n_parents: 選択する親の数
            
        Returns:
            選択された親モデルのリスト
        """
        parents = []
        
        # 適合度の合計
        total_fitness = sum(fitness_scores)
        
        # 適合度が全て0の場合はランダム選択
        if total_fitness == 0:
            return random.sample(population, n_parents)
        
        # 選択確率の計算
        probabilities = [f / total_fitness for f in fitness_scores]
        
        # 親選択
        indices = np.random.choice(
            range(len(population)),
            size=n_parents,
            p=probabilities,
            replace=False
        )
        
        for idx in indices:
            parents.append(population[idx])
        
        return parents
    
    def calculate_fitness(
        self,
        performances: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> List[float]:
        """適合度の計算
        
        Args:
            performances: 各モデルのパフォーマンス指標
            metrics: 使用する評価指標（指定がなければ設定ファイルの値を使用）
            
        Returns:
            各モデルの適合度スコアのリスト
        """
        if metrics is None:
            metrics = self.learning_config.fitness_metrics
        
        fitness_scores = []
        
        for perf in performances:
            # 各評価指標の値を正規化して合計
            metric_values = []
            
            for metric in metrics:
                if metric in perf:
                    value = perf[metric]
                    
                    # 無効な値の処理
                    if value is None or np.isnan(value) or np.isinf(value):
                        value = 0
                    
                    metric_values.append(value)
                else:
                    # 指標がない場合は0
                    metric_values.append(0)
            
            # 適合度スコアの計算（単純な合計）
            # より洗練された計算方法に拡張可能
            fitness = sum(metric_values)
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def evolve_generation(
        self,
        current_population: List[Dict[str, Any]],
        performances: List[Dict[str, Any]],
        hyperparameter_space: Dict[str, Any],
        all_features: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """世代の進化
        
        Args:
            current_population: 現在の世代のモデル集団
            performances: 各モデルのパフォーマンス評価
            hyperparameter_space: ハイパーパラメータの探索空間
            all_features: 全ての利用可能な特徴量リスト
            
        Returns:
            次世代のモデル集団
        """
        # 適合度の計算
        fitness_scores = self.calculate_fitness(performances)
        
        # エリート選択（上位n個をそのまま次世代に残す）
        elite_size = self.learning_config.elite_size
        population_size = len(current_population)
        
        # 適合度でソート
        sorted_indices = np.argsort(fitness_scores)[::-1]  # 降順
        elites = [current_population[i] for i in sorted_indices[:elite_size]]
        
        # 残りの集団を生成
        remaining_size = population_size - elite_size
        next_generation = elites.copy()
        
        # 交叉による子の生成
        for _ in range(remaining_size // 2):
            # 親選択
            parents = self.select_parents(
                current_population,
                fitness_scores,
                2,
                method="tournament"
            )
            
            # 交配
            if random.random() < self.learning_config.crossover_rate:
                child = self.crossover(
                    parents[0],
                    parents[1],
                    hyperparameter_space
                )
            else:
                # 交配しない場合は親の一方をコピー
                child = copy.deepcopy(random.choice(parents))
                child["uuid"] = str(uuid.uuid4())
                child["parent_uuid"] = [p["uuid"] for p in parents]
            
            # 突然変異
            if random.random() < self.learning_config.mutation_rate:
                child = self.mutate(
                    child,
                    hyperparameter_space,
                    all_features=all_features
                )
            
            next_generation.append(child)
        
        # 足りない場合は突然変異で補完
        while len(next_generation) < population_size:
            parent = random.choice(current_population)
            mutated = self.mutate(
                parent,
                hyperparameter_space,
                mutation_rate=0.3,  # 高い突然変異率
                all_features=all_features
            )
            next_generation.append(mutated)
        
        # 世代情報の更新
        for i, model in enumerate(next_generation):
            if i >= elite_size:  # エリート以外
                model["generation"] = max(m["generation"] for m in current_population) + 1
                model["name"] = f"{model['model_type']}_{model['currency_pair']}_g{model['generation']}_{i}"
        
        return next_generation 