import torch

model = torch.load("htc++_beitv2_adapter_large_fpn_o365.pth")  # backbone
model1 = torch.load("co_detr_head_pretrain.pth")  # head


state_dict = {}
for k, v in model["state_dict"].items():
    if 'backbone' in k:
        print(k)
        state_dict[k] = v

for k, v in model1.items():
    state_dict[k] = v

for k in state_dict.keys():
    print(k)


torch.save(state_dict, "o365_backbone_codetr_head.pth")


# model = torch.load("o365_backbone_codetr_head.pth")  # backbone

# for k, v in model.items():
#     print(k)


'''

missing keys in source state_dict: backbone.level_embed, backbone.patch_embed.proj.weight, backbone.patch_embed.proj.bias, backbone.blocks.0.gamma_1, backbone.blocks.0.gamma_2, backbone.blocks.0.norm1.weight, backbone.blocks.0.norm1.bias, backbone.blocks.0.attn.q_bias, backbone.blocks.0.attn.v_bias, backbone.blocks.0.attn.relative_position_bias_table, backbone.blocks.0.attn.relative_position_index, backbone.blocks.0.attn.qkv.weight, backbone.blocks.0.attn.proj.weight, backbone.blocks.0.attn.proj.bias, backbone.blocks.0.norm2.weight, backbone.blocks.0.norm2.bias, backbone.blocks.0.mlp.fc1.weight, backbone.blocks.0.mlp.fc1.bias, backbone.blocks.0.mlp.fc2.weight, backbone.blocks.0.mlp.fc2.bias, backbone.blocks.1.gamma_1, backbone.blocks.1.gamma_2, backbone.blocks.1.norm1.weight, backbone.blocks.1.norm1.bias, backbone.blocks.1.attn.q_bias, backbone.blocks.1.attn.v_bias, backbone.blocks.1.attn.relative_position_bias_table, backbone.blocks.1.attn.relative_position_index, backbone.blocks.1.attn.qkv.weight, backbone.blocks.1.attn.proj.weight, backbone.blocks.1.attn.proj.bias, backbone.blocks.1.norm2.weight, backbone.blocks.1.norm2.bias, backbone.blocks.1.mlp.fc1.weight, backbone.blocks.1.mlp.fc1.bias, backbone.blocks.1.mlp.fc2.weight, backbone.blocks.1.mlp.fc2.bias, backbone.blocks.2.gamma_1, backbone.blocks.2.gamma_2, backbone.blocks.2.norm1.weight, backbone.blocks.2.norm1.bias, backbone.blocks.2.attn.q_bias, backbone.blocks.2.attn.v_bias, backbone.blocks.2.attn.relative_position_bias_table, backbone.blocks.2.attn.relative_position_index, backbone.blocks.2.attn.qkv.weight, backbone.blocks.2.attn.proj.weight, backbone.blocks.2.attn.proj.bias, backbone.blocks.2.norm2.weight, backbone.blocks.2.norm2.bias, backbone.blocks.2.mlp.fc1.weight, backbone.blocks.2.mlp.fc1.bias, backbone.blocks.2.mlp.fc2.weight, backbone.blocks.2.mlp.fc2.bias, backbone.blocks.3.gamma_1, backbone.blocks.3.gamma_2, backbone.blocks.3.norm1.weight, backbone.blocks.3.norm1.bias, backbone.blocks.3.attn.q_bias, backbone.blocks.3.attn.v_bias, backbone.blocks.3.attn.relative_position_bias_table, backbone.blocks.3.attn.relative_position_index, backbone.blocks.3.attn.qkv.weight, backbone.blocks.3.attn.proj.weight, backbone.blocks.3.attn.proj.bias, backbone.blocks.3.norm2.weight, backbone.blocks.3.norm2.bias, backbone.blocks.3.mlp.fc1.weight, backbone.blocks.3.mlp.fc1.bias, backbone.blocks.3.mlp.fc2.weight, backbone.blocks.3.mlp.fc2.bias, backbone.blocks.4.gamma_1, backbone.blocks.4.gamma_2, backbone.blocks.4.norm1.weight, backbone.blocks.4.norm1.bias, backbone.blocks.4.attn.q_bias, backbone.blocks.4.attn.v_bias, backbone.blocks.4.attn.relative_position_bias_table, backbone.blocks.4.attn.relative_position_index, backbone.blocks.4.attn.qkv.weight, backbone.blocks.4.attn.proj.weight, backbone.blocks.4.attn.proj.bias, backbone.blocks.4.norm2.weight, backbone.blocks.4.norm2.bias, backbone.blocks.4.mlp.fc1.weight, backbone.blocks.4.mlp.fc1.bias, backbone.blocks.4.mlp.fc2.weight, backbone.blocks.4.mlp.fc2.bias, backbone.blocks.5.gamma_1, backbone.blocks.5.gamma_2, backbone.blocks.5.norm1.weight, backbone.blocks.5.norm1.bias, backbone.blocks.5.attn.q_bias, backbone.blocks.5.attn.v_bias, backbone.blocks.5.attn.relative_position_bias_table, backbone.blocks.5.attn.relative_position_index, backbone.blocks.5.attn.qkv.weight, backbone.blocks.5.attn.proj.weight, backbone.blocks.5.attn.proj.bias, backbone.blocks.5.norm2.weight, backbone.blocks.5.norm2.bias, backbone.blocks.5.mlp.fc1.weight, backbone.blocks.5.mlp.fc1.bias, backbone.blocks.5.mlp.fc2.weight, backbone.blocks.5.mlp.fc2.bias, backbone.blocks.6.gamma_1, backbone.blocks.6.gamma_2, backbone.blocks.6.norm1.weight, backbone.blocks.6.norm1.bias, backbone.blocks.6.attn.q_bias, backbone.blocks.6.attn.v_bias, backbone.blocks.6.attn.relative_position_bias_table, backbone.blocks.6.attn.relative_position_index, backbone.blocks.6.attn.qkv.weight, backbone.blocks.6.attn.proj.weight, backbone.blocks.6.attn.proj.bias, backbone.blocks.6.norm2.weight, backbone.blocks.6.norm2.bias, backbone.blocks.6.mlp.fc1.weight, backbone.blocks.6.mlp.fc1.bias, backbone.blocks.6.mlp.fc2.weight, backbone.blocks.6.mlp.fc2.bias, backbone.blocks.7.gamma_1, backbone.blocks.7.gamma_2, backbone.blocks.7.norm1.weight, backbone.blocks.7.norm1.bias, backbone.blocks.7.attn.q_bias, backbone.blocks.7.attn.v_bias, backbone.blocks.7.attn.relative_position_bias_table, backbone.blocks.7.attn.relative_position_index, backbone.blocks.7.attn.qkv.weight, backbone.blocks.7.attn.proj.weight, backbone.blocks.7.attn.proj.bias, backbone.blocks.7.norm2.weight, backbone.blocks.7.norm2.bias, backbone.blocks.7.mlp.fc1.weight, backbone.blocks.7.mlp.fc1.bias, backbone.blocks.7.mlp.fc2.weight, backbone.blocks.7.mlp.fc2.bias, backbone.blocks.8.gamma_1, backbone.blocks.8.gamma_2, backbone.blocks.8.norm1.weight, backbone.blocks.8.norm1.bias, backbone.blocks.8.attn.q_bias, backbone.blocks.8.attn.v_bias,
'''