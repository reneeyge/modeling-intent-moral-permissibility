
import math

def logit(x, alpha_1, alpha_2):
    ins = -alpha_1 * (x-alpha_2)
    return (1 + math.exp(ins))**(-1)


def intent_perm(p_harm, alpha_1=7, alpha_2=0.7):
    """
    p_harm (float): p(intent_harm = Yes | action A)
    """
    return logit((1 - p_harm), alpha_1, alpha_2)


def util_perm(lost_lives, saved_lives, alpha_1=0.3, alpha_2=0):
    net = lost_lives - saved_lives
    return logit(net, alpha_1, alpha_2)


def full_perm(p_harm, lost_lives, saved_lives, w=0.8):
    per_intent = intent_perm(p_harm)
    per_util = util_perm(lost_lives, saved_lives)
    per_full = w * per_intent + (1 - w) * per_util
    return per_full


# applying model
p_harm_data = [0.033821777, 0.255182789, 0.082900495, 0.277950774, 0.162973162, 0.30249444, 0.067307358, 0.243347983, 0.076269175, 0.444233919, 0.081700562]
lives = [(1, 5), (1, 5), (1, 2), (1, 2), (1, 1), (1, 1), (1, 1), (2, 1), (2, 1), (5, 1), (5, 1)]
for i in range(len(lives)):
    p_harm = p_harm_data[i]
    lives_lost = lives[i][0]
    lives_saved = lives[i][1]
    print(full_perm(p_harm, lives_lost, lives_saved, 0.8))