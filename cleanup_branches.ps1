# Cleanup Script - Run AFTER changing default branch to 'master' on GitHub

# Delete the main branch (run this after changing default branch on GitHub)
echo "Deleting main branch from GitHub..."
git push origin --delete main

# Verify only master branch remains
echo "Remaining branches:"
git ls-remote --heads origin

echo "✅ Cleanup complete! Only 'master' branch should remain."
echo "🎯 Your enhanced RAG application is now the sole branch!"
