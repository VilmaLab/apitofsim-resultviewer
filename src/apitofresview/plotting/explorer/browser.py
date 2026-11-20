"""Small helpers for Explorer browser callbacks."""

ON_FRAME = """
function onFrame(owner, key, update) {
    if (owner[key]) return;
    owner[key] = requestAnimationFrame(() => {
        owner[key] = null;
        update();
    });
}
"""
