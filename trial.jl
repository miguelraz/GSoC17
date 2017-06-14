using Lint

str2 = """
    test = "a"
    kj
    """;

msgs2 = lintfile("file",str2)
