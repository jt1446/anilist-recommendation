'''Inputs: gnn_model, seq_model, processed_data

Outputs: trained_weights.pth, loss_history.csv

Key Functions:

    bpr_loss(pos_scores, neg_scores): Implements the Bayesian Personalized Ranking loss.

    train_step(gnn, rnn, batch_data, optimizer): Coordinates the forward pass through both models and the backward pass for gradients.

    evaluate_hit_rate(model, test_data, k=10): Calculates your primary success metric.
'''
