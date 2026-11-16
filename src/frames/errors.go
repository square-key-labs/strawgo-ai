package frames

import (
	"context"
	"errors"
	"net"
)

type ErrorCategory string

const (
	ErrorCategoryUnknown         ErrorCategory = "unknown"
	ErrorCategoryAuthentication  ErrorCategory = "authentication"
	ErrorCategoryAuthorization   ErrorCategory = "authorization"
	ErrorCategoryInvalidRequest  ErrorCategory = "invalid_request"
	ErrorCategoryRateLimit       ErrorCategory = "rate_limit"
	ErrorCategoryQuota           ErrorCategory = "quota"
	ErrorCategoryConnectivity    ErrorCategory = "connectivity"
	ErrorCategoryServer          ErrorCategory = "server"
	ErrorCategoryApplication     ErrorCategory = "application"
	ErrorCategoryContentFiltered ErrorCategory = "content_filtered"
)

func (c ErrorCategory) IsPermanent() bool {
	switch c {
	case ErrorCategoryAuthentication, ErrorCategoryAuthorization, ErrorCategoryInvalidRequest:
		return true
	}
	return false
}

func ClassifyHTTPStatus(code int) ErrorCategory {
	switch code {
	case 400, 404, 422:
		return ErrorCategoryInvalidRequest
	case 401:
		return ErrorCategoryAuthentication
	case 402:
		return ErrorCategoryQuota
	case 403:
		return ErrorCategoryAuthorization
	case 429:
		return ErrorCategoryRateLimit
	}
	if code >= 500 && code < 600 {
		return ErrorCategoryServer
	}
	return ErrorCategoryUnknown
}

func ClassifyError(err error) ErrorCategory {
	if err == nil || errors.Is(err, context.Canceled) {
		return ErrorCategoryUnknown
	}
	var categorized interface{ ErrorCategory() ErrorCategory }
	if errors.As(err, &categorized) {
		if c := categorized.ErrorCategory(); c != "" {
			return c
		}
	}
	var status interface{ HTTPStatusCode() int }
	if errors.As(err, &status) {
		return ClassifyHTTPStatus(status.HTTPStatusCode())
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return ErrorCategoryConnectivity
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return ErrorCategoryConnectivity
	}
	return ErrorCategoryUnknown
}
