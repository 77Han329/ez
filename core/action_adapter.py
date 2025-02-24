import collections
from config.atari import game_config

class ActionAdapter:
    def __init__(self):
        """
        Action Adapter M: Maps real actions a_t to latent action embeddings e_k
        """
        model = game_config.get_uniform_network(True)
        self.lag_model = model.lag  # 预训练 LAG
        self.mapping = {}  # M 映射表
        self.count_table = collections.defaultdict(lambda: collections.defaultdict(int))  # C[a, k] 计数表

    def update_count_table(self, s_t, a_t, s_t1):
        """
        use LAG to calculate e_k and update C[a, k]
        """
        _, _, _, e_k = self.lag_model(s_t, s_t1)  # latent action embedding indicies
        self.count_table[a_t][e_k.item()] += 1  # update C[a, k]

    def build_adapter(self):
        """
        使用 C 构建 Action Adapter M
        """
        sorted_table = []
        for a_t in self.count_table.keys():
            sorted_ek = sorted(self.count_table[a_t].items(), key=lambda x: -x[1])
            sorted_table.append((a_t, sorted_ek))

        self.mapping.clear()
        used_ek = set()

        for a_t, sorted_ek in sorted_table:
            for e_k, count in sorted_ek:
                if a_t not in self.mapping and e_k not in used_ek:
                    self.mapping[a_t] = e_k
                    used_ek.add(e_k)

        return self.mapping  

    def get_latent_action(self, a_t):
        """
        获取 a_t 对应的 e_k
        """
        return self.mapping.get(a_t, None)
