import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    width: 1500
    height: 940
    minimumWidth: 1120
    minimumHeight: 700
    visible: true
    title: "VIctory — Grand Theft Auto VI"
    color: "#090b0e"
    font.family: "Segoe UI"

    property int pageIndex: 0
    readonly property color paper: "#f0ece7"
    readonly property color muted: "#aaa39b"
    readonly property color red: "#cf293c"

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.preferredWidth: 82
            Layout.fillHeight: true
            color: "#090b0e"
            border.color: "#292b2e"
            border.width: 1

            ColumnLayout {
                anchors.fill: parent
                anchors.topMargin: 40
                anchors.bottomMargin: 24
                spacing: 0

                Rectangle {
                    Layout.alignment: Qt.AlignHCenter
                    width: 52
                    height: 58
                    radius: 8
                    color: window.pageIndex === 0 ? "#21171a" : "#131518"
                    Text {
                        anchors.centerIn: parent
                        text: "VI"
                        color: "#f0c5a1"
                        font.family: "Georgia"
                        font.pixelSize: 23
                    }
                    Rectangle {
                        visible: window.pageIndex === 0
                        width: 3
                        height: 46
                        anchors.left: parent.left
                        anchors.verticalCenter: parent.verticalCenter
                        color: window.red
                    }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: window.pageIndex = 0
                    }
                }
                Rectangle {
                    Layout.alignment: Qt.AlignHCenter
                    Layout.topMargin: 16
                    width: 48
                    height: 1
                    color: "#292b2e"
                }
                Repeater {
                    model: [
                        { label: "Library", glyph: "⌂", page: 0 },
                        { label: "Activity", glyph: "◷", page: 1 },
                        { label: "Settings", glyph: "⚙", page: 2 }
                    ]
                    delegate: Item {
                        required property var modelData
                        Layout.alignment: Qt.AlignHCenter
                        Layout.topMargin: 12
                        width: 52
                        height: 54
                        Rectangle {
                            anchors.fill: parent
                            radius: 7
                            color: window.pageIndex === modelData.page ? "#21171a" : navMouse.containsMouse ? "#191a1d" : "transparent"
                        }
                        Rectangle {
                            visible: window.pageIndex === modelData.page
                            width: 3
                            height: 38
                            anchors.left: parent.left
                            anchors.verticalCenter: parent.verticalCenter
                            color: window.red
                        }
                        Text {
                            anchors.centerIn: parent
                            text: modelData.glyph
                            color: window.pageIndex === modelData.page ? "#f0c5a1" : "#8b8985"
                            font.pixelSize: 23
                        }
                        ToolTip {
                            visible: navMouse.containsMouse
                            text: modelData.label
                            delay: 450
                        }
                        MouseArea {
                            id: navMouse
                            anchors.fill: parent
                            hoverEnabled: true
                            onClicked: window.pageIndex = modelData.page
                        }
                    }
                }
                Item { Layout.fillHeight: true }
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "VI"
                    color: "#5a5957"
                    font.family: "Georgia"
                    font.pixelSize: 11
                }
            }
        }

        Item {
            id: stage
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            Image {
                anchors.fill: parent
                source: launcher.wallpaperSource
                fillMode: Image.PreserveAspectCrop
                sourceSize.width: 3840
                sourceSize.height: 2160
                asynchronous: true
                smooth: true
            }
            Rectangle {
                anchors.fill: parent
                gradient: Gradient {
                    orientation: Gradient.Horizontal
                    GradientStop { position: 0.0; color: "#f208090c" }
                    GradientStop { position: 0.39; color: "#ed080a0d" }
                    GradientStop { position: 0.64; color: "#a90a0c10" }
                    GradientStop { position: 1.0; color: "#2b080a0c" }
                }
            }
            Rectangle {
                anchors.fill: parent
                gradient: Gradient {
                    GradientStop { position: 0.0; color: "#17050709" }
                    GradientStop { position: 0.52; color: "#10050709" }
                    GradientStop { position: 1.0; color: "#8505070a" }
                }
            }

            StackLayout {
                anchors.fill: parent
                currentIndex: window.pageIndex

                Flickable {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.leftMargin: 42
                    Layout.rightMargin: 30
                    Layout.topMargin: 29
                    Layout.bottomMargin: 22
                    contentWidth: width
                    contentHeight: homeContent.implicitHeight
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    ColumnLayout {
                        id: homeContent
                        width: Math.min(parent.width, 665)
                        spacing: 0
                        RowLayout {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 34
                            spacing: 13
                            Text {
                                text: "GRAND THEFT AUTO VI"
                                color: window.paper
                                font.family: "Georgia"
                                font.pixelSize: 16
                                font.letterSpacing: 2.1
                            }
                            Text { text: "/"; color: window.red; font.pixelSize: 17 }
                            Text {
                                text: "LAUNCHER"
                                color: "#b3a69e"
                                font.pixelSize: 12
                                font.letterSpacing: 2.3
                            }
                            Item { Layout.fillWidth: true }
                            Rectangle {
                                width: 8
                                height: 8
                                radius: 4
                                color: launcher.running ? "#e1b16d" : launcher.gameExists ? "#82bd91" : "#d39c70"
                            }
                            Text {
                                text: launcher.running ? "In game" : launcher.gameExists ? "Ready" : "Setup"
                                color: "#c7c3bd"
                                font.pixelSize: 12
                            }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.topMargin: 10
                            Layout.bottomMargin: 30
                            height: 1
                            color: "#54c0c3c5"
                        }
                        Text {
                            Layout.fillWidth: true
                            text: launcher.gameExists ? "Ready to play" : "Welcome to Leonida"
                            color: window.paper
                            font.family: "Georgia"
                            font.pixelSize: Math.min(58, Math.max(43, stage.width * 0.041))
                            font.weight: Font.Normal
                            elide: Text.ElideRight
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.topMargin: 18
                            Layout.bottomMargin: 19
                            height: 1
                            color: "#54c0c3c5"
                        }
                        Text {
                            Layout.fillWidth: true
                            text: "VICE CITY   ·   SERIES " + (launcher.consoleModel === "Series X" ? "X" : "S") + "   ·   " + (launcher.gameExists ? "GAME READY" : "LOCAL LIBRARY")
                            color: "#eee8e2"
                            font.family: "Georgia"
                            font.pixelSize: 17
                            elide: Text.ElideRight
                        }
                        Text {
                            Layout.fillWidth: true
                            Layout.topMargin: 8
                            Layout.bottomMargin: 23
                            text: launcher.gameExists ? launcher.selectedPath : "Your next story starts in Vice City. Add your GTA VI game files to continue."
                            color: "#b1aba4"
                            font.pixelSize: 12
                            elide: Text.ElideMiddle
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 62
                            radius: 7
                            color: launchMouse.containsMouse && !launcher.running ? "#e43a4d" : window.red
                            opacity: launcher.running ? 0.68 : 1
                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: 22
                                anchors.rightMargin: 22
                                spacing: 12
                                Text { text: launcher.running ? "●" : "▶"; color: "white"; font.pixelSize: 15 }
                                Text {
                                    Layout.fillWidth: true
                                    horizontalAlignment: Text.AlignHCenter
                                    text: launcher.running ? "GAME IN SESSION" : launcher.gameExists ? "PLAY GRAND THEFT AUTO VI" : launcher.hasGame ? "LOCATE GAME FILES" : "ADD GAME FILES"
                                    color: "white"
                                    font.family: "Georgia"
                                    font.pixelSize: 18
                                    elide: Text.ElideRight
                                }
                                Item { width: 15; height: 1 }
                            }
                            MouseArea {
                                id: launchMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                enabled: !launcher.running
                                onClicked: launcher.gameExists ? launcher.launch() : launcher.browse()
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            Layout.topMargin: 14
                            Layout.bottomMargin: 9
                            spacing: 8
                            Text {
                                text: "CONSOLE PROFILE"
                                color: "#a49d96"
                                font.pixelSize: 10
                                font.weight: Font.DemiBold
                                font.letterSpacing: 1.4
                            }
                            Item { Layout.fillWidth: true }
                            Repeater {
                                model: ["Series X", "Series S"]
                                delegate: Rectangle {
                                    required property string modelData
                                    Layout.preferredWidth: 105
                                    Layout.preferredHeight: 34
                                    radius: 6
                                    color: launcher.consoleModel === modelData ? "#a72635" : "#a00d1014"
                                    border.color: launcher.consoleModel === modelData ? "#df4858" : "#4b4d50"
                                    Text {
                                        anchors.centerIn: parent
                                        text: modelData
                                        color: launcher.consoleModel === modelData ? "#fff5f1" : "#c6c0b9"
                                        font.pixelSize: 12
                                    }
                                    MouseArea {
                                        anchors.fill: parent
                                        onClicked: launcher.setConsoleModel(modelData)
                                    }
                                }
                            }
                        }
                        Text {
                            Layout.fillWidth: true
                            Layout.bottomMargin: 15
                            text: launcher.consoleModel + " profile selected for your next session."
                            color: "#aaa39b"
                            font.pixelSize: 11
                        }
                        Repeater {
                            model: [
                                { title: "Game files & library", detail: launcher.hasGame ? "Grand Theft Auto VI  ·  Ready" : "Choose your GTA VI executable", glyph: "▣", action: "files" },
                                { title: "Launch activity", detail: launcher.logs.length > 0 ? launcher.logs[launcher.logs.length - 1] : "Your recent local sessions", glyph: "◷", action: "activity" },
                                { title: "Console profile", detail: launcher.consoleModel + "  ·  Change target", glyph: "⌖", action: "profile" },
                                { title: "Launcher settings", detail: "Local data and app information", glyph: "⚙", action: "settings" }
                            ]
                            delegate: Rectangle {
                                required property var modelData
                                Layout.fillWidth: true
                                Layout.preferredHeight: 60
                                Layout.bottomMargin: 9
                                radius: 7
                                color: cardMouse.containsMouse ? "#c01b1d21" : "#a0121417"
                                border.color: modelData.action === "files" && !launcher.hasGame ? window.red : "#37393d"
                                RowLayout {
                                    anchors.fill: parent
                                    anchors.leftMargin: 19
                                    anchors.rightMargin: 16
                                    spacing: 16
                                    Text {
                                        text: modelData.glyph
                                        color: "#d1cbc4"
                                        font.pixelSize: 20
                                        Layout.preferredWidth: 20
                                        horizontalAlignment: Text.AlignHCenter
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.title
                                        color: window.paper
                                        font.family: "Georgia"
                                        font.pixelSize: 17
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        Layout.preferredWidth: 170
                                        horizontalAlignment: Text.AlignRight
                                        text: modelData.detail
                                        color: "#afa69d"
                                        font.pixelSize: 11
                                        elide: Text.ElideRight
                                    }
                                    Text { text: "›"; color: "#d6d0c8"; font.pixelSize: 24 }
                                }
                                MouseArea {
                                    id: cardMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    onClicked: {
                                        if (modelData.action === "files") launcher.browse()
                                        else if (modelData.action === "activity") window.pageIndex = 1
                                        else window.pageIndex = 2
                                    }
                                }
                            }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.topMargin: 17
                            Layout.bottomMargin: 10
                            height: 1
                            color: "#4ac0c3c5"
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: launcher.status
                                color: "#a49d96"
                                font.pixelSize: 11
                                elide: Text.ElideRight
                                Layout.fillWidth: true
                            }
                            Text {
                                text: "VICTORY  ·  WINDOWS"
                                color: "#85817c"
                                font.pixelSize: 10
                                font.letterSpacing: 1.0
                            }
                        }
                    }
                }

                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 48
                        anchors.rightMargin: 48
                        anchors.topMargin: 40
                        anchors.bottomMargin: 34
                        spacing: 16
                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: "ROCKSTAR GAMES   /   ACTIVITY"
                                color: window.paper
                                font.family: "Georgia"
                                font.pixelSize: 18
                                font.letterSpacing: 1.4
                            }
                            Item { Layout.fillWidth: true }
                            Rectangle {
                                width: 144
                                height: 38
                                radius: 6
                                color: clearMouse.containsMouse ? "#252629" : "#17191c"
                                border.color: "#4b4d50"
                                Text {
                                    anchors.centerIn: parent
                                    text: "CLEAR ACTIVITY"
                                    color: "#d8d2ca"
                                    font.pixelSize: 11
                                    font.weight: Font.DemiBold
                                    font.letterSpacing: 1.0
                                }
                                MouseArea {
                                    id: clearMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    onClicked: launcher.clearActivity()
                                }
                            }
                        }
                        Text {
                            text: "Local launch history for Grand Theft Auto VI."
                            color: window.muted
                            font.pixelSize: 13
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            radius: 8
                            color: "#d20d0f12"
                            border.color: "#47484a"
                            ListView {
                                id: activityList
                                anchors.fill: parent
                                anchors.margins: 20
                                spacing: 12
                                clip: true
                                model: launcher.logs
                                onCountChanged: positionViewAtEnd()
                                delegate: RowLayout {
                                    required property string modelData
                                    required property int index
                                    width: activityList.width
                                    spacing: 13
                                    Rectangle {
                                        width: 7
                                        height: 7
                                        radius: 4
                                        color: index === activityList.count - 1 ? window.red : "#8b8883"
                                        Layout.alignment: Qt.AlignVCenter
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData
                                        color: "#ded8d1"
                                        font.family: "Cascadia Mono"
                                        font.pixelSize: 12
                                        wrapMode: Text.Wrap
                                    }
                                }
                                Text {
                                    anchors.centerIn: parent
                                    visible: activityList.count === 0
                                    text: "Your GTA VI launch activity will appear here."
                                    color: "#b5aea6"
                                    font.family: "Georgia"
                                    font.pixelSize: 19
                                }
                            }
                        }
                    }
                }

                Item {
                    Flickable {
                        anchors.fill: parent
                        anchors.margins: 46
                        contentWidth: width
                        contentHeight: settingsContent.implicitHeight
                        clip: true
                        ColumnLayout {
                            id: settingsContent
                            width: parent.width
                            spacing: 13
                            Text {
                                text: "GRAND THEFT AUTO VI   /   SETTINGS"
                                color: window.paper
                                font.family: "Georgia"
                                font.pixelSize: 19
                                font.letterSpacing: 1.2
                            }
                            Text {
                                Layout.bottomMargin: 8
                                text: "Launcher profile, game location, and local data."
                                color: window.muted
                                font.pixelSize: 13
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 108
                                radius: 7
                                color: "#d20d0f12"
                                border.color: "#47484a"
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 8
                                    Text {
                                        text: "CONSOLE PROFILE"
                                        color: "#d4a9a0"
                                        font.pixelSize: 10
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.5
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Text {
                                            Layout.fillWidth: true
                                            text: "New sessions use " + launcher.consoleModel + "."
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 17
                                        }
                                        Repeater {
                                            model: ["Series X", "Series S"]
                                            delegate: Rectangle {
                                                required property string modelData
                                                width: 105
                                                height: 36
                                                radius: 6
                                                color: launcher.consoleModel === modelData ? "#a72635" : "#202225"
                                                border.color: launcher.consoleModel === modelData ? "#df4858" : "#47494c"
                                                Text {
                                                    anchors.centerIn: parent
                                                    text: modelData
                                                    color: "#f2ede7"
                                                    font.pixelSize: 12
                                                }
                                                MouseArea {
                                                    anchors.fill: parent
                                                    onClicked: launcher.setConsoleModel(modelData)
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 143
                                radius: 7
                                color: "#d20d0f12"
                                border.color: "#47484a"
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 8
                                    Text {
                                        text: "GAME FILES"
                                        color: "#d4a9a0"
                                        font.pixelSize: 10
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.5
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: launcher.hasGame ? launcher.selectedPath : "Select the GTA VI executable in your existing game folder."
                                        color: window.paper
                                        font.pixelSize: 12
                                        wrapMode: Text.WrapAnywhere
                                        maximumLineCount: 2
                                        elide: Text.ElideMiddle
                                    }
                                    Rectangle {
                                        Layout.preferredWidth: 196
                                        Layout.preferredHeight: 35
                                        radius: 5
                                        color: browseSettingsMouse.containsMouse ? "#292b2e" : "#202225"
                                        border.color: "#484a4d"
                                        Text {
                                            anchors.centerIn: parent
                                            text: launcher.hasGame ? "CHANGE GAME LOCATION" : "LOCATE GTA VI"
                                            color: "#eee8e2"
                                            font.pixelSize: 10
                                            font.weight: Font.DemiBold
                                            font.letterSpacing: 1
                                        }
                                        MouseArea {
                                            id: browseSettingsMouse
                                            anchors.fill: parent
                                            hoverEnabled: true
                                            onClicked: launcher.browse()
                                        }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 123
                                radius: 7
                                color: "#d20d0f12"
                                border.color: "#47484a"
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 8
                                    Text {
                                        text: "LOCAL LAUNCHER DATA"
                                        color: "#d4a9a0"
                                        font.pixelSize: 10
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.5
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: launcher.dataDirectory
                                        color: window.paper
                                        font.family: "Cascadia Mono"
                                        font.pixelSize: 11
                                        wrapMode: Text.Wrap
                                    }
                                    Text {
                                        text: "VIctory keeps launcher preferences and activity on this PC."
                                        color: window.muted
                                        font.pixelSize: 11
                                    }
                                }
                            }
                            Text {
                                Layout.topMargin: 8
                                text: "VIctory for Windows  ·  PySide6 / Qt Quick"
                                color: "#95908a"
                                font.pixelSize: 11
                            }
                        }
                    }
                }
            }
        }
    }
}
