package awsauth

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
)

const DefaultRegion = "us-east-1"

type Config struct {
	AWSConfig       *aws.Config
	Region          string
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
	Profile         string
	Endpoint        string
	HTTPClient      config.HTTPClient
}

func Load(ctx context.Context, c Config) (aws.Config, error) {
	if c.AWSConfig != nil {
		cfg := c.AWSConfig.Copy()
		if c.Region != "" {
			cfg.Region = c.Region
		}
		if c.AccessKeyID != "" && c.SecretAccessKey != "" {
			cfg.Credentials = aws.NewCredentialsCache(credentials.NewStaticCredentialsProvider(c.AccessKeyID, c.SecretAccessKey, c.SessionToken))
		}
		if c.HTTPClient != nil {
			cfg.HTTPClient = c.HTTPClient
		}
		if cfg.Region == "" {
			cfg.Region = DefaultRegion
		}
		return cfg, nil
	}
	var opts []func(*config.LoadOptions) error
	if c.Region != "" {
		opts = append(opts, config.WithRegion(c.Region))
	}
	if c.Profile != "" {
		opts = append(opts, config.WithSharedConfigProfile(c.Profile))
	}
	if c.AccessKeyID != "" && c.SecretAccessKey != "" {
		opts = append(opts, config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(c.AccessKeyID, c.SecretAccessKey, c.SessionToken)))
	}
	if c.HTTPClient != nil {
		opts = append(opts, config.WithHTTPClient(c.HTTPClient))
	}
	cfg, err := config.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return aws.Config{}, err
	}
	if cfg.Region == "" {
		cfg.Region = DefaultRegion
	}
	return cfg, nil
}

func (c Config) BaseEndpoint() *string {
	if c.Endpoint == "" {
		return nil
	}
	return aws.String(c.Endpoint)
}
