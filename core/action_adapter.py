# test implementation of action adapter and couting table


# from config.atari import game_config
# import torch
# class ActionAdapter:
#     def __init__(self):
        
#         model = game_config.get_uniform_network(True)
#         self.lag_model = model.lag 

#         # {(a, k): count}
#         self.count_table = {}

#         # a -> e_k
#         self.action_adapter = {}

#     def update_count_table(self, s_t, a_t, s_t1):
#         self.lag_model.eval()
#         with torch.no_grad():
#             _, _, _, encoding_indices = self.lag_model(s_t, s_t1)   

#         key = (a_t, encoding_indices)
#         self.count_table[key] = self.count_table.get(key, 0) + 1

#     def build_adapter(self):
#         self.action_adapter.clear()
#         table_list = [
#             (a, latent_idx, count)
#             for ((a, latent_idx), count) in self.count_table.items()
#         ]
        
#         table_list.sort(key=lambda x: x[2], reverse=True)

        
#         for (a, latent_idx, count) in table_list:
            
#             if a in self.action_adapter:
#                 continue
            
#             if latent_idx in self.action_adapter.values():
#                 continue

#             self.action_adapter[a] = latent_idx

#     def get_latent_action(self, a_t):
        
#         idx = self.get_latent_index(a_t)
#         if idx is None:
#             return None
        
#         embedding_vector = self.lag_model.quantizer.get_embedding(idx)
#         return embedding_vector
