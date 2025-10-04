package loss

// import (
// 	"fmt"

// 	"github.com/gocnn/spark"
// )

// // NLL computes the negative log likelihood loss.
// //
// // Arguments:
// //   - inp: Input tensor of dimensions [N, C] where N is batch size and C is number of categories.
// //     Expected to contain log probabilities.
// //   - target: Ground truth labels as tensor of uint32 of dimension [N].
// //
// // Returns a scalar tensor containing the average value over the batch.
// func NLL[T spark.D](inp, target *spark.Tensor[T]) (*spark.Tensor[T], error) {
// 	inpShape := inp.Layout().Shape()
// 	targetShape := target.Layout().Shape()

// 	// Validate target shape: should be [batch_size]
// 	if targetShape.Rank() != 1 {
// 		return nil, fmt.Errorf("target tensor should have a single dimension, got %v", targetShape)
// 	}
// 	batchSize := targetShape.Dim(0)

// 	// Validate input shape: should be [batch_size, num_classes]
// 	if inpShape.Rank() != 2 {
// 		return nil, fmt.Errorf("input tensor should have two dimensions, got %v", inpShape)
// 	}
// 	if inpShape.Dim(0) != batchSize {
// 		return nil, fmt.Errorf("batch size mismatch between input (%d) and target (%d)",
// 			inpShape.Dim(0), batchSize)
// 	}

// 	// Gather operation: inp.gather(target.unsqueeze(1), dim=1)
// 	targetUnsqueezed, err := target.Unsqueeze(1)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to unsqueeze target: %w", err)
// 	}

// 	gathered, err := inp.Gather(targetUnsqueezed, 1)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to gather: %w", err)
// 	}

// 	// Sum all elements
// 	summed, err := gathered.SumAll()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to sum: %w", err)
// 	}

// 	// Apply affine transformation: -1/batch_size * sum + 0
// 	scale := -1.0 / float64(batchSize)
// 	result, err := summed.Affine(scale, 0.0)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to apply affine transformation: %w", err)
// 	}

// 	return result, nil
// }

// // CrossEntropy computes the cross-entropy loss.
// //
// // Arguments:
// //   - inp: Input tensor of dimensions [N, C] where N is batch size and C is number of categories.
// //     Expected to contain raw logits.
// //   - target: Ground truth labels as tensor of uint32 of dimension [N].
// //
// // Returns a scalar tensor containing the average value over the batch.
// func CrossEntropy[T spark.D](inp, target *spark.Tensor[T]) (*spark.Tensor[T], error) {
// 	if inp.Layout().Shape().Rank() != 2 {
// 		return nil, fmt.Errorf("cross_entropy expects an input tensor of rank 2, got %d",
// 			inp.Layout().Shape().Rank())
// 	}

// 	// Apply log_softmax to input along dimension 1
// 	logSoftmax, err := inp.LogSoftmax(1)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute log_softmax: %w", err)
// 	}

// 	// Compute NLL loss
// 	return NLL(logSoftmax, target)
// }

// // MSE computes the mean squared error loss.
// //
// // Arguments:
// // - inp: Input tensor
// // - target: Target tensor (same shape as input)
// //
// // Returns a scalar tensor containing the mean squared error.
// func MSE[T spark.D](inp, target *spark.Tensor[T]) (*spark.Tensor[T], error) {
// 	// Compute (inp - target)
// 	diff, err := inp.Sub(target)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute difference: %w", err)
// 	}

// 	// Square the difference
// 	squared, err := diff.Sqr()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to square: %w", err)
// 	}

// 	// Compute mean over all elements
// 	mean, err := squared.MeanAll()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute mean: %w", err)
// 	}

// 	return mean, nil
// }

// // BinaryCrossEntropyWithLogits computes the binary cross-entropy loss with logits.
// //
// // Arguments:
// //   - inp: Input tensor of dimensions [N, C] where N is batch size and C is number of categories.
// //     Expected to contain raw logits.
// //   - target: Ground truth labels as tensor of dimension [N, C] where N is batch size and C is number of categories.
// //
// // Returns a scalar tensor containing the average value over the batch.
// func BinaryCrossEntropyWithLogits[T spark.D](inp, target *spark.Tensor[T]) (*spark.Tensor[T], error) {
// 	// Apply sigmoid to input
// 	sigmoid, err := inp.Sigmoid()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute sigmoid: %w", err)
// 	}

// 	// Compute log of sigmoid
// 	sigmoidLog, err := sigmoid.Log()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute log of sigmoid: %w", err)
// 	}

// 	// Left side: target * log(sigmoid(inp))
// 	leftSide, err := target.Mul(sigmoidLog)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute left side: %w", err)
// 	}

// 	// Compute (1 - target)
// 	oneMinusTarget, err := target.Affine(-1.0, 1.0)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute (1 - target): %w", err)
// 	}

// 	// Compute (1 - sigmoid(inp))
// 	oneMinusSigmoid, err := sigmoid.Affine(-1.0, 1.0)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute (1 - sigmoid): %w", err)
// 	}

// 	// Compute log(1 - sigmoid(inp))
// 	oneMinusSigmoidLog, err := oneMinusSigmoid.Log()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute log(1 - sigmoid): %w", err)
// 	}

// 	// Right side: (1 - target) * log(1 - sigmoid(inp))
// 	rightSide, err := oneMinusTarget.Mul(oneMinusSigmoidLog)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute right side: %w", err)
// 	}

// 	// Combine: left_side + right_side
// 	combined, err := leftSide.Add(rightSide)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to combine sides: %w", err)
// 	}

// 	// Negate the result
// 	negated, err := combined.Neg()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to negate: %w", err)
// 	}

// 	// Compute mean over all elements
// 	loss, err := negated.MeanAll()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute mean: %w", err)
// 	}

// 	return loss, nil
// }

// // L1Loss computes the L1 (Mean Absolute Error) loss.
// //
// // Arguments:
// // - inp: Input tensor
// // - target: Target tensor (same shape as input)
// //
// // Returns a scalar tensor containing the mean absolute error.
// func L1Loss[T spark.D](inp, target *spark.Tensor[T]) (*spark.Tensor[T], error) {
// 	// Compute (inp - target)
// 	diff, err := inp.Sub(target)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute difference: %w", err)
// 	}

// 	// Take absolute value
// 	abs, err := diff.Abs()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute absolute value: %w", err)
// 	}

// 	// Compute mean over all elements
// 	mean, err := abs.MeanAll()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute mean: %w", err)
// 	}

// 	return mean, nil
// }

// // SmoothL1Loss computes the Smooth L1 loss (Huber loss).
// //
// // Arguments:
// // - inp: Input tensor
// // - target: Target tensor (same shape as input)
// // - beta: Threshold for switching between L1 and L2 loss (default: 1.0)
// //
// // Returns a scalar tensor containing the smooth L1 loss.
// func SmoothL1Loss[T spark.D](inp, target *spark.Tensor[T], beta float64) (*spark.Tensor[T], error) {
// 	if beta <= 0 {
// 		beta = 1.0
// 	}

// 	// Compute absolute difference
// 	diff, err := inp.Sub(target)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute difference: %w", err)
// 	}

// 	absDiff, err := diff.Abs()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute absolute difference: %w", err)
// 	}

// 	// Create condition: |diff| < beta
// 	betaTensor, err := absDiff.ZerosLike()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to create beta tensor: %w", err)
// 	}
// 	betaTensor, err = betaTensor.AddScalar(beta)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to add beta: %w", err)
// 	}

// 	condition, err := absDiff.Lt(betaTensor)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute condition: %w", err)
// 	}

// 	// L2 part: 0.5 * diff^2 / beta
// 	diffSqr, err := diff.Sqr()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to square difference: %w", err)
// 	}
// 	l2Part, err := diffSqr.MulScalar(0.5 / beta)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute L2 part: %w", err)
// 	}

// 	// L1 part: |diff| - 0.5 * beta
// 	l1Part, err := absDiff.AddScalar(-0.5 * beta)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute L1 part: %w", err)
// 	}

// 	// Select between L2 and L1 based on condition
// 	result, err := condition.Where(l2Part, l1Part)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to select loss: %w", err)
// 	}

// 	// Compute mean
// 	mean, err := result.MeanAll()
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to compute mean: %w", err)
// 	}

// 	return mean, nil
// }
