import kagglehub

# Download latest version
path = kagglehub.competition_download('data')

print("Path to competition files:", path)