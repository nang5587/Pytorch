def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=100,
    device="cpu",
):
    """모델 훈련 함수"""
    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    patience = 20  # 조기 종료를 위한 patience

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

        # 검증 모드
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()

        # 평균 계산
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * correct_train / total_train
        val_acc = 100.0 * correct_val / total_val

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # 학습률 스케줄러
        if scheduler:
            scheduler.step(avg_val_loss)

        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            if (epoch + 1) % 10 == 0:
                print(
                    f"=*=*=*= Validation Loss decreased to {avg_val_loss:.6f}. Saving the model! =*=*=*="
                )
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

        # 조기 종료
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 최고 성능 모델 로드
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

# 손실함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=10, factor=0.5
)

# 5. 모델 훈련
print(f"\n5. 모델 훈련 시작...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

history = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=300,
    device=device,
)
